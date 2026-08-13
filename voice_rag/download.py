from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx


async def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[download] {destination} already exists, skipping")
        return
    print(f"[download] fetching {url}")
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(destination, "wb") as fh:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r[download] {done / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB", end="")
    print(f"\n[download] saved to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Hindi MSMARCO-XI validation split")
    parser.add_argument("--url", default="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet")
    parser.add_argument("--path", type=Path, default=Path("data/raw/hinval.parquet"))
    args = parser.parse_args()
    asyncio.run(download(args.url, args.path))


if __name__ == "__main__":
    main()
