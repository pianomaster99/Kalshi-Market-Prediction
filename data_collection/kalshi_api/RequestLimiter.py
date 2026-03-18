from dataclasses import dataclass
import asyncio
from typing import Any, Optional, Dict, List
import time
import requests
from collections import deque

from data_collection.kalshi_api.utils import BASE_URL


@dataclass
class RestRequest:
    method: str
    path: str
    params: Optional[Dict[str, Any]] = None
    json: Optional[Dict[str, Any]] = None
    timeout: float = 20.0


@dataclass
class Job:
    req: RestRequest
    future: asyncio.Future


class RequestLimiter:
    def __init__(self, limit_per_second: int, max_queue: int = 2000):
        if limit_per_second <= 0:
            raise ValueError("limit_per_second must be > 0")

        self.limit_per_second = limit_per_second
        self.q: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_queue)
        self.session = requests.Session()

        self._tasks: List[asyncio.Task] = []
        self._closed = False

        # Global (shared) sliding-window state across ALL workers
        self._timestamps = deque()
        self._rate_lock = asyncio.Lock()

    async def start(self, workers: int = 10):
        if self._tasks:
            return
        self._closed = False
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(workers)]

    async def close(self):
        self._closed = True

        # Cancel workers
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self.session.close()

    async def request(self, req: RestRequest) -> requests.Response:
        if self._closed:
            raise RuntimeError("RequestLimiter is closed")

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self.q.put(Job(req=req, future=fut))
        return await fut

    def _request(self, req: RestRequest) -> requests.Response:
        url = f"{BASE_URL}{req.path}"
        method = req.method.upper()

        if method == "GET":
            return self.session.get(url, params=req.params, timeout=req.timeout)
        elif method == "POST":
            return self.session.post(url, params=req.params, json=req.json, timeout=req.timeout)
        else:
            raise ValueError(f"Unsupported method: {req.method}")

    async def _acquire_rate_slot(self):
        # Sliding window: max N requests in the last 1 second, shared across workers
        while True:
            async with self._rate_lock:
                now = time.monotonic()

                while self._timestamps and (now - self._timestamps[0] > 1.0):
                    self._timestamps.popleft()

                if len(self._timestamps) < self.limit_per_second:
                    self._timestamps.append(now)
                    return

                wait = 1.0 - (now - self._timestamps[0])

            # sleep outside lock
            await asyncio.sleep(max(0.0, wait))

    async def _worker(self, worker_id: int):
        try:
            while True:
                job = await self.q.get()
                try:
                    await self._acquire_rate_slot()
                    resp = await asyncio.to_thread(self._request, job.req)
                    if not job.future.done():
                        job.future.set_result(resp)
                except Exception as e:
                    if not job.future.done():
                        job.future.set_exception(e)
                finally:
                    self.q.task_done()
        except asyncio.CancelledError:
            # exit cleanly
            return