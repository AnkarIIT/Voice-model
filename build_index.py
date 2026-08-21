import argparse
import time
from pathlib import Path

import polars as pl

from app.chunking import hybrid_chunk_row
from app.config import (
    EMBED_MODEL,
    INDEX_DIR,
    SAMPLE_PARQUET,
    WHISPER_DEVICE,
    setup_logging,
)
from app.embed import FaissIndex
from app.retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from sample parquet")
    parser.add_argument("--rows", type=int, default=800, help="number of dataset rows to index")
    parser.add_argument("--data", type=Path, default=SAMPLE_PARQUET)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    parser.add_argument("--device", default=WHISPER_DEVICE)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.data} rows={args.rows}")
    df = pl.read_parquet(args.data).head(args.rows)
    print(f"rows {df.height}")

    from sentence_transformers import SentenceTransformer

    sem_model = SentenceTransformer(EMBED_MODEL, device=args.device)

    all_chunks = []
    t0 = time.perf_counter()
    for row in df.iter_rows(named=True):
        all_chunks.extend(hybrid_chunk_row(row, sem_model))
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Chunking done: {len(all_chunks)} chunks from {df.height} rows in {build_ms:.1f}ms")
    for s in ["fixed256_overlap20", "fixed512_overlap15", "semantic", "metadata_raw"]:
        print(f" {s}: {sum(1 for c in all_chunks if c['strategy'] == s)}")

    idx = FaissIndex(EMBED_MODEL, args.device)
    print(f"Embedding {len(all_chunks)} chunks dim={idx.dim} device={idx.device} gpu={idx.gpu}")
    emb_ms = idx.add(all_chunks, batch_size=128)
    print(f"Embed+Add {emb_ms:.1f}ms ntotal={idx.index.ntotal}")
    idx.save(args.out)

    qcol = "query" if "query" in df.columns else None
    if not args.skip_bench and qcol:
        run_smoke(df, idx)


def run_smoke(df, idx):
    from app.retrieval import retrieve as _retrieve  # noqa: F401

    row0 = df.row(0, named=True)
    q = row0.get("query") or row0.get("Eng_Query") or ""
    if not isinstance(q, str) or len(q) < 3:
        return
    res = retrieve(q, idx, k=5)
    print(f"\nQuery: {q[:120].encode('ascii', 'ignore').decode()}")
    print(f"search_ms {res['search_ms']} total {res['total_ms']}")
    for r in res["results"]:
        print(
            f" [{r['score']:.3f} {r['strategy']}] "
            f"{r['text'][:120].encode('ascii', 'ignore').decode()}"
        )

    queries = [
        r["query"]
        for r in df.head(50).iter_rows(named=True)
        if isinstance(r.get("query"), str) and len(r["query"]) > 5
    ][:30]
    times = sorted(idx.search(qq, 5)[1] for qq in queries)

    def pct(p):
        return times[min(int(len(times) * p / 100), len(times) - 1)] if times else 0

    print(
        f"\nLatency over {len(times)} queries (embedding+FAISS): "
        f"P50={pct(50):.2f}ms P70={pct(70):.2f}ms P100={max(times):.2f}ms"
    )
    print(f"Target 80-100ms: {'PASS' if pct(50) < 100 else 'TUNE NEEDED'}")


if __name__ == "__main__":
    setup_logging()
    main()
