import asyncio
import gzip
import json
from typing import Iterator

SENTINEL = object()   # signal to stop the writer

async def file_write(queue: asyncio.Queue, filename: str):
    with gzip.open(filename, "at", encoding="utf-8") as f:
        while True:
            item = await queue.get()

            if item is SENTINEL:
                queue.task_done()
                break

            try:
                f.write(item + "\n")

            finally:
                queue.task_done()

def file_read(path: str) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line