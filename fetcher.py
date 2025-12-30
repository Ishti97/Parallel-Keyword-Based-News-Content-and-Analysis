
import aiohttp
import asyncio
import requests

async def fetch(session, url):
    """Fetch a single URL with error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(url, timeout=timeout) as r:
            return url, await r.text()
    except Exception as e:
        print(f"      [WARN] Failed to fetch {url}: {type(e).__name__}")
        return url, ""

async def fetch_all_async(urls):
    """Fetch all URLs concurrently using async I/O."""
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch(session, u) for u in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

def fetch_all_sequential(urls):
    """Fetch all URLs sequentially (one at a time)."""
    results = []
    for u in urls:
        try:
            r = requests.get(u, timeout=15)
            results.append((u, r.text))
        except Exception as e:
            print(f"      [WARN] Failed to fetch {u}: {type(e).__name__}")
            results.append((u, ""))
    return results
