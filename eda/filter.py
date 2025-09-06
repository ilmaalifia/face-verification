"""Filter valid image URLs from facescrub_metadata.csv and save to filtered_facescrub_metadata.csv"""

import asyncio

import aiohttp
import pandas as pd

df = pd.read_csv("data/facescrub_metadata.csv")
sem = asyncio.Semaphore(30)


async def check_url(session, url):
    try:
        async with sem:
            async with session.head(url, timeout=5, allow_redirects=True) as head_resp:
                if head_resp.status // 100 == 2:
                    return url

                # Fallback if HEAD fails
                async with session.get(
                    url, timeout=5, allow_redirects=True
                ) as get_resp:
                    if get_resp.status // 100 == 2:
                        return url
    except:
        pass
    return None


async def filter_urls(df):
    print(f"Total URLs: {len(df)}")
    async with aiohttp.ClientSession() as session:
        tasks = [check_url(session, url) for url in df["url"]]
        results = await asyncio.gather(*tasks)
    return df[df["url"].isin([r for r in results if r])]


if __name__ == "__main__":
    valid_df = asyncio.run(filter_urls(df))
    print(f"Valid URLs: {len(valid_df)}")
    valid_df.to_csv("data/filtered_facescrub_metadata.csv", index=False)
