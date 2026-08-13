from __future__ import annotations

import re
from typing import Optional

import numpy as np

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokeniser: lowercases and keeps letters/digits.

    Works for both Latin and Devanagari scripts; drops 1-char and very long
    tokens (URLs / mojibake fragments).
    """
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if 1 < len(tok) < 30:
            out.append(tok)
    return out


class BM25Index:
    """Lucene-style Okapi BM25 over a column-compressed term-count matrix.

    Built once at index time from the child-layer texts; query-time scoring is
    a few sparse column lookups, so a hybrid call adds ~1ms.
    """

    def __init__(
        self,
        vocab: list[str],
        idf: np.ndarray,
        rows: np.ndarray,
        data: np.ndarray,
        col_ptr: np.ndarray,
        doc_len: np.ndarray,
        n_docs: int,
        avgdl: float,
    ) -> None:
        self.vocab = vocab
        self.idf = idf
        self.rows = rows
        self.data = data
        self.col_ptr = col_ptr
        self.doc_len = doc_len
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.vocab_id = {tok: i for i, tok in enumerate(vocab)}

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        n_docs = len(texts)
        vocab: dict[str, int] = {}
        doc_len = np.zeros(n_docs, dtype=np.int32)
        all_rows: list[int] = []
        all_cols: list[int] = []
        all_data: list[int] = []
        for doc_idx, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            doc_len[doc_idx] = sum(counts.values())
            for tok, count in counts.items():
                tid = vocab.setdefault(tok, len(vocab))
                all_rows.append(doc_idx)
                all_cols.append(tid)
                all_data.append(count)

        cols = np.asarray(all_cols, dtype=np.int32)
        rows = np.asarray(all_rows, dtype=np.int32)
        data = np.asarray(all_data, dtype=np.int32)
        order = np.argsort(cols, kind="stable")
        rows = rows[order]
        data = data[order]
        col_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.add.at(col_ptr, cols[order] + 1, 1)
        np.cumsum(col_ptr, out=col_ptr)

        df = np.diff(col_ptr).astype(np.float32)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(doc_len.mean()) if n_docs else 1.0
        vocab_list = [None] * len(vocab)
        for tok, tid in vocab.items():
            vocab_list[tid] = tok
        return cls(vocab_list, idf, rows, data, col_ptr, doc_len, n_docs, avgdl)

    def scores(self, query: str) -> np.ndarray:
        arr = np.zeros(self.n_docs, dtype=np.float32)
        for tok in tokenize(query):
            tid = self.vocab_id.get(tok)
            if tid is None:
                continue
            start = int(self.col_ptr[tid])
            end = int(self.col_ptr[tid + 1])
            if end <= start:
                continue
            tfs = self.data[start:end].astype(np.float32)
            doc_ids = self.rows[start:end]
            dls = self.doc_len[doc_ids].astype(np.float32)
            denom = tfs + K1 * (1.0 - B + B * dls / self.avgdl)
            np.add.at(arr, doc_ids, self.idf[tid] * (tfs * (K1 + 1.0)) / denom)
        return arr

    def top(self, query: str, k: int) -> list[tuple[float, int]]:
        scores = self.scores(query)
        if scores.size == 0:
            return []
        order = np.argpartition(scores, -min(k, scores.size))[-k:]
        return [(float(scores[i]), int(i)) for i in order[np.argsort(-scores[order])]]

    def pack(self) -> dict:
        return {
            "bm25_rows": self.rows,
            "bm25_data": self.data,
            "bm25_col_ptr": self.col_ptr,
            "bm25_idf": self.idf,
            "bm25_doc_len": self.doc_len,
            "bm25_n_docs": self.n_docs,
            "bm25_avgdl": self.avgdl,
            "bm25_vocab": self.vocab,
        }

    @classmethod
    def unpack(cls, arrays: dict, meta: dict) -> Optional["BM25Index"]:
        if "bm25_vocab" not in meta:
            return None
        return cls(
            vocab=meta["bm25_vocab"],
            idf=arrays["bm25_idf"],
            rows=arrays["bm25_rows"],
            data=arrays["bm25_data"],
            col_ptr=arrays["bm25_col_ptr"],
            doc_len=arrays["bm25_doc_len"],
            n_docs=int(arrays["bm25_n_docs"]),
            avgdl=float(arrays["bm25_avgdl"]),
        )
