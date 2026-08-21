import os
import sys

os.environ["PYTHONUTF8"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from app.preprocess import clean_text

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
REPO = "ai4bharat/MSMARCO-XI"
TARGETS = [("validation/hinval.parquet", 5000), ("validation/benval.parquet", 2000)]
TEXT_COLS = ["query", "Answer", "Eng_Query", "Eng_Answer"]


def main():
    for hf_path, sample_n in TARGETS:
        print(f"\n[download] {hf_path}")
        local = hf_hub_download(
            repo_id=REPO, repo_type="dataset", filename=hf_path, local_dir=DATA_DIR
        )
        print(f" -> {local} size={Path(local).stat().st_size / 1e6:.1f} MB")
        df = pl.read_parquet(local)
        print(f" rows={df.height} cols={df.columns}")
        sample = df.head(sample_n)
        for c in TEXT_COLS:
            if c in sample.columns:
                sample = sample.with_columns(
                    pl.col(c).map_elements(clean_text, return_dtype=pl.String).alias(c)
                )
        out = DATA_DIR / f"sample_{Path(hf_path).stem}_{sample_n}.parquet"
        sample.write_parquet(out)
        print(f" [sample] wrote {out} rows={sample.height}")
        row = sample.row(0, named=True)
        print(
            f" cols present ok, query_id={row.get('query_id')} type={row.get('query_type')}"
        )

    print("\nDone.")
    for p in sorted(DATA_DIR.rglob("*.parquet")):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
