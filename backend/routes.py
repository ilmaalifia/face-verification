from typing import List, Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from backend.dependencies.facenet import facenet_model
from backend.dependencies.milvus import milvus

router = APIRouter()


class VerifyResponse(BaseModel):
    is_duplicate: Optional[bool] = None
    results: Optional[object] = None  # Duplicate results
    embedding: Optional[List[float]] = None
    error: Optional[str] = None


@router.post("/verify/", response_model=VerifyResponse)
async def verify(img_file_buffer: UploadFile = File(...)):
    image_bytes = await img_file_buffer.read()
    if not image_bytes:
        return VerifyResponse(
            is_duplicate=None, results=None, error="Empty file uploaded"
        )
    img = facenet_model.read_image(image_bytes)
    embeddings = facenet_model.get_embeddings(img)
    embedding = embeddings[0]["embedding"]
    results = milvus.search_data(embedding)
    if results and len(results[0]) > 0:
        # Convert Milvus search results to a JSON serializable format
        final_results = []
        for result in results[0]:
            final_results.append(
                {
                    "id": result.id,
                    "distance": result.distance,
                    "entity": {
                        "name": result.entity.get("name"),
                        "file_path": result.entity.get("file_path"),
                    },
                }
            )
        return VerifyResponse(
            is_duplicate=True, results=final_results, embedding=embedding, error=None
        )
    else:
        return VerifyResponse(
            is_duplicate=False, results=[], embedding=embedding, error=None
        )
