import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Due to TF1 and TF2 mismatch


class Settings(BaseSettings):
    MILVUS_URI: str = os.getenv("MILVUS_URI", "./milvus_demo.db")
    MILVUS_TOKEN: Optional[str] = os.getenv("MILVUS_TOKEN")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "face_embeddings")
    MILVUS_RADIUS: float = float(os.getenv("MILVUS_RADIUS", "1.0"))
    MILVUS_RANGE_FILTER: float = float(os.getenv("MILVUS_RANGE_FILTER", "0.0"))
    TOP_K: int = int(os.getenv("TOP_K", "7"))
    LOCAL_IMG_DIR: str = "/img"


settings = Settings()
