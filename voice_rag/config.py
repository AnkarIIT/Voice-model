from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    elevenlabs_api_key: str = ""

    elevenlabs_stt_model: str = "scribe_v1"
    groq_model: str = "llama-3.1-8b-instant"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    data_dir: Path = Path("data")
    index_name: str = "msmarco_hi_val_v1"
    index_parquet: Path = Path("data/raw/hinval.parquet")
    index_max_rows: int = 1500
    index_embed_workers: int = 6
    index_seed: int = 42

    top_k: int = 3
    parent_search_top_k: int = 3
    child_search_top_k: int = 6
    retrieval_score_threshold: float = 0.28
    off_topic_score_threshold: float = 0.20

    grounding_lexical_threshold: float = 0.10
    grounding_embedding_threshold: float = 0.45

    max_answer_tokens: int = 96
    llm_temperature: float = 0.0
    llm_json_mode: bool = True

    max_retries: int = 3
    retry_base_delay_s: float = 0.3
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: int = 30
    llm_timeout_s: int = 10
    stt_timeout_s: int = 20

    benchmark_n_queries: int = 120
    benchmark_seed: int = 7

    mock_mode: bool = True

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / self.index_name

    @property
    def metrics_dir(self) -> Path:
        return self.data_dir / "metrics"


def get_settings() -> Settings:
    return Settings()
