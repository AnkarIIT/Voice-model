import hashlib
import re
from typing import Dict, List

from .preprocess import clean_text, normalize_passages


def fixed_size_overlap(text: str, chunk_tokens: int = 512, overlap: float = 0.2) -> List[str]:
    if chunk_tokens < 20:
        raise ValueError("chunk_tokens must be >= 20")
    overlap = min(max(overlap, 0.0), 0.9)
    words = text.split()
    if len(words) <= chunk_tokens:
        return [text] if text.strip() else []
    step = max(1, int(chunk_tokens * (1 - overlap)))
    chunks: List[str] = []
    i = 0
    while i < len(words):
        part = words[i : i + chunk_tokens]
        if len(part) >= 20:
            chunks.append(" ".join(part))
        if i + chunk_tokens >= len(words):
            break
        i += step
    return chunks


def semantic_split(
    text: str, model=None, threshold: float = 0.65, max_sent_per_chunk: int = 4
) -> List[str]:
    sents = re.split(r"(?<=[.!?।])\s+", text)
    sents = [clean_text(s) for s in sents if len(s.strip()) > 10]
    if len(sents) <= 1:
        return [s for s in sents if s]
    if model is None:
        return [
            " ".join(sents[i : i + max_sent_per_chunk])
            for i in range(0, len(sents), max_sent_per_chunk)
        ]
    import numpy as np

    embs = model.encode(sents, normalize_embeddings=True)
    chunks = []
    cur = [sents[0]]
    for i in range(1, len(sents)):
        sim = float(np.dot(embs[i - 1], embs[i]))
        if sim < threshold or len(cur) >= max_sent_per_chunk:
            chunks.append(" ".join(cur))
            cur = [sents[i]]
        else:
            cur.append(sents[i])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def metadata_aware_split(row: dict, use_translated: bool = True) -> List[Dict]:
    passages = normalize_passages(row, use_translated)
    for p in passages:
        p["meta_label"] = f"{p['query_type']}|{p['target_lang']}|doc:{p['doc_id']}"
    return passages


def hybrid_chunk_row(row: dict, embed_model=None) -> List[Dict]:
    metas = metadata_aware_split(row, True)
    all_chunks: List[Dict] = []
    for base in metas:
        txt = base["text"]
        for c in fixed_size_overlap(txt, 256, 0.2):
            all_chunks.append({**base, "text": c, "strategy": "fixed256_overlap20"})
        for c in fixed_size_overlap(txt, 512, 0.15):
            all_chunks.append({**base, "text": c, "strategy": "fixed512_overlap15"})
        for c in semantic_split(txt, embed_model, 0.68, 3):
            all_chunks.append({**base, "text": c, "strategy": "semantic"})
        all_chunks.append({**base, "strategy": "metadata_raw"})
    seen = set()
    uniq = []
    for c in all_chunks:
        key = (c["doc_id"], c["strategy"], hashlib.sha1(c["text"].encode("utf-8")).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq
