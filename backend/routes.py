import os
import random
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from backend.dependencies.facenet import facenet_model
from backend.dependencies.milvus import milvus
from backend.settings import settings

router = APIRouter()


class RegisterResponse(BaseModel):
    is_registered: bool
    error: Optional[str] = None


@router.post("/register/", response_model=RegisterResponse)
async def register(
    name: str = Form(...),
    is_agree: bool = Form(...),
    is_duplicate: bool = Form(...),
    embedding: List[float] = Form(...),
    img_file_buffer: UploadFile = File(...),
):
    if not is_agree:
        return RegisterResponse(is_registered=False, error="Terms not agreed")
    if is_duplicate:
        return RegisterResponse(is_registered=False, error="Duplicate image found")
    image_bytes = await img_file_buffer.read()
    if not image_bytes:
        return RegisterResponse(is_registered=False, error="Empty file uploaded")

    os.makedirs(settings.LOCAL_IMG_DIR, exist_ok=True)
    filename = f"{'_'.join(name.lower().split())}_{image_id}_{face_id}.jpg"
    file_path = os.path.join(settings.LOCAL_IMG_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    image_id = random.randint(10, 999999)  # Generate random image_id
    face_id = random.randint(10, 999999)  # Generate random face_id
    inserted = milvus.insert_data(
        {
            "image_id": image_id,
            "face_id": face_id,
            "name": name,
            "embedding": embedding,
            "file_path": file_path,
            "timestamp": int(time.time() * 1000),
        }
    )
    return (
        RegisterResponse(is_registered=True, error=None)
        if inserted and inserted.get("insert_count", 0) > 0
        else RegisterResponse(
            is_registered=False, error="Failed during database insertion"
        )
    )


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
