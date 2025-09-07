import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pandas as pd

from backend.dependencies.facenet import facenet_model
from backend.dependencies.milvus import milvus

sem = asyncio.Semaphore(10)
queue = asyncio.Queue(maxsize=100)


async def get_image(session, url, name, image_id, face_id):
    try:
        async with sem:
            async with session.get(url, timeout=5, allow_redirects=True) as get_resp:
                is_success = get_resp.status // 100 == 2
                content_type = get_resp.headers.get("Content-Type", "").lower()
                is_image = (
                    content_type.startswith("image/")
                    or content_type == "application/octet-stream"
                )
                if is_success and is_image:
                    img_file_buffer = await get_resp.read()
                    await queue.put((url, name, image_id, face_id, img_file_buffer))
    except:
        pass
    return None


def process_image(item):
    url, name, image_id, face_id, img_file_buffer = item
    try:
        img = facenet_model.read_image(img_file_buffer)
        embeddings = facenet_model.get_embeddings(img)
        inserted = milvus.insert_data(
            {
                "image_id": image_id,
                "face_id": face_id,
                "name": name,
                "embedding": embeddings[0]["embedding"],
                "file_path": url,
                "timestamp": int(time.time() * 1000),
            }
        )
        if inserted.get("insert_count", 0) > 0:
            print(f"Successfully inserted face embedding from {url}")
    except Exception as e:
        print(f"Processing failed {url}: {e}")


async def main(df):
    print(f"Total URLs: {len(df)}")
    counter = 0
    async with aiohttp.ClientSession() as session:
        # Get image file
        producers = []
        for row in df.itertuples(index=False):
            producers.append(
                get_image(session, row.url, row.name, row.image_id, row.face_id)
            )
        producer_task = asyncio.gather(*producers)

        # Generate embeddings
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as pool:
            while True:
                if producer_task.done() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=2)
                except asyncio.TimeoutError:
                    continue
                loop.run_in_executor(pool, process_image, item)
                counter += 1
                if counter % 100 == 0:
                    milvus.flush()

        await producer_task


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/facescrub_metadata.csv")
        valid_df = asyncio.run(main(df))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        milvus.get_info()
