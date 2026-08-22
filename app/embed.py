import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "cuda") -> str:
    if requested != "cuda":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    logger.warning("CUDA requested but unavailable; using CPU")
    return "cpu"


def _to_gpu(index):
    if faiss.get_num_gpus() > 0:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index), True
        except Exception as e:
            logger.warning("FAISS GPU move failed (%s); staying on CPU", e)
    return index, False


class FaissIndex:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.index, self.gpu = _to_gpu(self.index)
        self.meta = []

    @property
    def texts(self):
        return [m["text"] for m in self.meta]

    def add(self, chunks: list, batch_size: int = 256) -> float:
        texts = [c["text"] for c in chunks]
        t0 = time.perf_counter()
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embs = np.ascontiguousarray(embs, dtype="float32")
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.meta.extend(chunks)
        return (time.perf_counter() - t0) * 1000

    def search(self, query: str, k: int = 5) -> tuple:
        t0 = time.perf_counter()
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        q = np.ascontiguousarray(q, dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        dt = (time.perf_counter() - t0) * 1000
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
            results.append({**self.meta[idx], "score": float(score), "rank": len(results)})
        return results, dt

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.gpu else self.index
        faiss.write_index(cpu_index, str(path / "faiss.index"))
        payload = {
            "model_name": self.model_name,
            "dim": self.dim,
            "ntotal": int(cpu_index.ntotal),
            "meta": self.meta,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info("saved index ntotal=%d to %s", cpu_index.ntotal, path)

    @classmethod
    def load(cls, path: Path, device: str = "cuda"):
        meta_path = Path(path) / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                payload = json.load(f)
            model_name = payload["model_name"]
            meta = payload["meta"]
        else:
            import pickle

            legacy = Path(path) / "meta.pkl"
            logger.warning("meta.json missing; loading legacy %s (pickle)", legacy)
            with open(legacy, "rb") as f:
                meta, _texts, model_name = pickle.load(f)
        obj = cls(model_name, device)
        obj.index = faiss.read_index(str(Path(path) / "faiss.index"))
        obj.index, obj.gpu = _to_gpu(obj.index)
        obj.meta = meta
        if not meta_path.exists():
            try:
                payload = {
                    "model_name": model_name,
                    "dim": obj.dim,
                    "ntotal": int(obj.index.ntotal),
                    "meta": meta,
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                logger.info("migrated legacy metadata to %s", meta_path)
            except (OSError, TypeError) as e:
                logger.warning("could not migrate legacy meta to json: %s", e)
        logger.info("loaded index ntotal=%d gpu=%s from %s", obj.index.ntotal, obj.gpu, path)
        return obj
