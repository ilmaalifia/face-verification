import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    MILVUS_URI: str = os.getenv("MILVUS_URI", "data/milvus_demo.db")
    MILVUS_TOKEN: Optional[str] = os.getenv("MILVUS_TOKEN")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "face_embeddings")
    MILVUS_RADIUS: float = float(os.getenv("MILVUS_RADIUS", "1.0"))
    MILVUS_RANGE_FILTER: float = float(os.getenv("MILVUS_RANGE_FILTER", "0.0"))
    TOP_K: int = int(os.getenv("TOP_K", "7"))


settings = Settings()
