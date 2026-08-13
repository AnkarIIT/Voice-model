from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


@dataclass
class ChunkNode:
    text: str
    language: str = ""
    source_kind: str = ""
    source_query_id: int = -1
    selected: bool = False
    strategy: str = ""
    word_count: int = 0
    passage_idx: Optional[int] = None
    parent_id: Optional[str] = None
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = uuid.uuid4().hex[:12]


class FixedSizeChunker:
    """Word-window chunking with configurable overlap.

    Retained as a baseline strategy: sliding windows of ``window_words`` words,
    stepping by ``window_words * (1 - overlap)``. Results are merged into whole
    sentences at the edges to avoid cutting mid-sentence.
    """

    def __init__(self, window_words: int = 200, overlap: float = 0.15) -> None:
        self.window_words = window_words
        self.overlap = overlap

    def chunk(self, text: str, **meta) -> list[ChunkNode]:
        words = _WORD_RE.findall(text)
        if not words:
            return []
        step = max(1, int(self.window_words * (1 - self.overlap)))
        chunks: list[ChunkNode] = []
        seen = 0
        for start in range(0, max(1, len(words) - step + 1), step):
            end = min(len(words), start + self.window_words)
            piece = " ".join(words[start:end])
            node = ChunkNode(
                text=piece,
                strategy="fixed_size",
                word_count=end - start,
                **meta,
            )
            if node.text != chunks[-1].text if chunks else True:
                chunks.append(node)
            seen += step
            if end >= len(words):
                break
        return chunks


class SemanticChunker:
    """Sentence-boundary chunking driven by embedding similarity.

    Chunk boundaries fall where the cosine similarity between adjacent sentences
    drops below ``merge_threshold`` (unless the running chunk is below
    ``min_words``). ``overlap_sentences`` carries the trailing sentences of the
    previous chunk into the next one so cross-boundary context is preserved.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        target_words: int = 96,
        min_words: int = 24,
        max_words: int = 192,
        merge_threshold: float = 0.45,
        overlap_sentences: int = 0,
    ) -> None:
        self.embed_fn = embed_fn
        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words
        self.merge_threshold = merge_threshold
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, **meta) -> list[ChunkNode]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            wc = word_count(sentences[0])
            if wc <= self.max_words:
                return [
                    ChunkNode(
                        text=sentences[0],
                        strategy="semantic",
                        word_count=wc,
                        **meta,
                    )
                ]
            words = sentences[0].split()
            step = max(1, self.max_words // 2)
            nodes: list[ChunkNode] = []
            for start in range(0, max(1, len(words) - step + 1), step):
                end = min(len(words), start + self.max_words)
                nodes.append(
                    ChunkNode(
                        text=" ".join(words[start:end]),
                        strategy="semantic_oversize",
                        word_count=end - start,
                        **meta,
                    )
                )
                if end >= len(words):
                    break
            return nodes

        embs = np.asarray(self.embed_fn(sentences), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        sims = (embs[:-1] * embs[1:]).sum(axis=1)

        nodes: list[ChunkNode] = []
        cur: list[int] = []
        cur_words = 0
        start_of_run = 0
        n = len(sentences)
        i = 0
        while i < n:
            if not cur:
                cur = [i]
                cur_words = word_count(sentences[i])
                start_of_run = i
                i += 1
                continue
            w = word_count(sentences[i])
            sim = sims[i - 1]
            under_min = cur_words < self.min_words
            can_merge = cur_words + w <= self.max_words
            keep_going = (under_min or sim >= self.merge_threshold) and can_merge
            hit_target = cur_words >= self.target_words and not under_min
            if keep_going and not (hit_target and sim < self.merge_threshold):
                cur.append(i)
                cur_words += w
                i += 1
            else:
                text_piece = " ".join(sentences[j] for j in cur)
                nodes.append(
                    ChunkNode(
                        text=text_piece,
                        strategy="semantic",
                        word_count=cur_words,
                        **meta,
                    )
                )
                if self.overlap_sentences and len(cur) > self.overlap_sentences:
                    cur = cur[-self.overlap_sentences:]
                    cur_words = sum(word_count(sentences[j]) for j in cur)
                else:
                    cur = []
                    cur_words = 0
        if cur:
            text_piece = " ".join(sentences[j] for j in cur)
            nodes.append(
                ChunkNode(
                    text=text_piece,
                    strategy="semantic",
                    word_count=cur_words,
                    **meta,
                )
            )
        return nodes


class ParentChildChunker:
    """Two-tier parent-child chunking.

    Parents are wide fixed-size windows (large context for generation).
    Each parent is re-split with the :class:`SemanticChunker` into fine-grained
    children (fast, precise retrieval). Children carry ``parent_id`` and inherit
    the parent text for context reconstruction.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        parent_words: int = 500,
        parent_overlap: float = 0.1,
        child_target_words: int = 90,
        child_min_words: int = 20,
        child_max_words: int = 160,
        child_overlap_sentences: int = 1,
        child_merge_threshold: float = 0.45,
    ) -> None:
        self.embed_fn = embed_fn
        self.parent_chunker = FixedSizeChunker(parent_words, parent_overlap)
        self.child_chunker = SemanticChunker(
            embed_fn=embed_fn,
            target_words=child_target_words,
            min_words=child_min_words,
            max_words=child_max_words,
            merge_threshold=child_merge_threshold,
            overlap_sentences=child_overlap_sentences,
        )

    def chunk(self, text: str, **meta) -> tuple[list[ChunkNode], list[ChunkNode]]:
        total_words = word_count(text)
        if total_words == 0:
            return [], []
        if total_words <= self.child_chunker.max_words:
            node = ChunkNode(
                text=text,
                strategy="atomic",
                word_count=total_words,
                **meta,
            )
            node.parent_id = node.chunk_id
            return [node], [node]

        parents = self.parent_chunker.chunk(text, **meta)
        children: list[ChunkNode] = []
        for parent in parents:
            parent.parent_id = parent.chunk_id
            subs = self.child_chunker.chunk(parent.text, **meta)
            for sub in subs:
                sub.parent_id = parent.chunk_id
                sub.strategy = f"semantic_in_parent:{sub.strategy}"
            children.extend(subs)
        return parents, children


def adaptive_chunk(
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    **meta,
) -> tuple[list[ChunkNode], list[ChunkNode]]:
    """Compositional strategy: atomic for short passages, parent-child otherwise.

    This is the entry point used by the indexer. It adapts the split to the
    document size, which is preferable to a single naive fixed-size strategy.
    """
    return ParentChildChunker(embed_fn=embed_fn).chunk(text, **meta)
