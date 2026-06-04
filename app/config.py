from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="K8S_RAG_",
        case_sensitive=False,
    )

    # Source
    k8s_website_repo: str = "https://github.com/kubernetes/website.git"
    k8s_docs_branch: str = "main"

    # Paths
    raw_docs_dir: Path = Path("data/raw/kubernetes-website")
    faiss_index_path: Path = Path("storage/faiss/index.faiss")
    sqlite_path: Path = Path("storage/sqlite/k8s_rag.db")

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 100
    max_chunk_size: int = 800

    # Retrieval
    top_k_initial: int = 20
    top_k_final: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation
    llm_backend: str = "openai_compatible"  # "openai_compatible" | "transformers"
    llm_model: str = "qwen3:8b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
