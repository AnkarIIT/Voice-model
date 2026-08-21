import re
import unicodedata


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_passages(row: dict, use_translated: bool = True) -> list:
    passages = row.get("passages") or {}
    eng = passages.get("English_passages") or []
    tra = passages.get("Translated_passages") or []
    sel = passages.get("is_selected") or []
    src = tra if use_translated else eng
    out = []
    for i, txt in enumerate(src):
        out.append(
            {
                "text": clean_text(txt),
                "doc_id": f"{row['query_id']}_{i}",
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "passage_idx": i,
                "is_selected": sel[i] if i < len(sel) else 0,
                "source_lang": row.get("source_lang", ""),
                "target_lang": row.get("target_lang", ""),
                "query": clean_text(row.get("query", "")),
            }
        )
    return out
