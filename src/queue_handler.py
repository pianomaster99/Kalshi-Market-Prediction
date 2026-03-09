import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from file_handler import FileWriter


SENTINEL = object()


@dataclass(frozen=True)
class QueueMessage:
    id: str
    message: str


class QueueHandler:
    def __init__(self, queue: asyncio.Queue, output_dir: str = "data", debug_every: int = 5000):
        self.queue = queue
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ids that are currently enabled for writing
        self._enabled_ids: set[str] = set()

        # id -> FileWriter
        self._writers: dict[str, FileWriter] = {}

        self._lock = asyncio.Lock()

        # debug counters
        self.msg_count = 0
        self.start_time = time.time()
        self.debug_every = debug_every

    def _sanitize_id(self, msg_id: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in msg_id)

    def _filename_for_id(self, msg_id: str) -> Path:
        return self.output_dir / f"{self._sanitize_id(msg_id)}.ndjson.gz"

    async def add_id(self, msg_id: str) -> None:
        async with self._lock:
            self._enabled_ids.add(msg_id)

    async def remove_id(self, msg_id: str) -> None:
        async with self._lock:
            self._enabled_ids.discard(msg_id)
            writer = self._writers.pop(msg_id, None)
            if writer is not None:
                writer.close()

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return sorted(self._enabled_ids)

    def _get_writer(self, msg_id: str) -> FileWriter:
        writer = self._writers.get(msg_id)
        if writer is None:
            filename = self._filename_for_id(msg_id)
            writer = FileWriter(str(filename))
            writer.open()
            self._writers[msg_id] = writer
        return writer

    def _close_all_writers(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()

    def _debug_status(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.msg_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[QueueHandler] {self.msg_count} msgs | "
            f"{rate:.1f} msg/s | queue={self.queue.qsize()} | "
            f"open_files={len(self._writers)} | enabled_ids={len(self._enabled_ids)}"
        )
        

    async def run(self) -> None:
        try:
            while True:
                item = await self.queue.get()
                try:
                    if item is SENTINEL:
                        break

                    if not isinstance(item, QueueMessage):
                        raise TypeError(
                            f"Expected QueueMessage or SENTINEL, got {type(item)!r}"
                        )

                    async with self._lock:
                        if item.id not in self._enabled_ids:
                            continue

                        writer = self._get_writer(item.id)
                        writer.write(item.message)

                    self.msg_count += 1
                    if self.msg_count % self.debug_every == 0:
                        self._debug_status()

                finally:
                    self.queue.task_done()
        finally:
            async with self._lock:
                self._close_all_writers()