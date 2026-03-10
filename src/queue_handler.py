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
    def __init__(
        self,
        queue: asyncio.Queue,
        output_dir: str = "data",
        debug_every: int = 5000,
        flush_every: int = 200,
    ):
        self.queue = queue
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._enabled_ids: set[str] = set()
        self._writers: dict[str, FileWriter] = {}
        self._lock = asyncio.Lock()

        self.msg_count = 0
        self.start_time = time.time()
        self.debug_every = debug_every
        self.flush_every = flush_every

    def _sanitize_id(self, msg_id: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in msg_id)

    def _filename_for_id(self, msg_id: str) -> Path:
        return self.output_dir / f"{self._sanitize_id(msg_id)}.ndjson.gz"

    async def add_id(self, msg_id: str) -> None:
        async with self._lock:
            self._enabled_ids.add(msg_id)

    async def remove_id(self, msg_id: str) -> None:
        writer = None
        async with self._lock:
            self._enabled_ids.discard(msg_id)
            writer = self._writers.pop(msg_id, None)

        if writer is not None:
            try:
                await asyncio.to_thread(writer.close_sync)
            except Exception as e:
                print(f"[QueueHandler] failed to close writer for {msg_id}: {e}")

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return sorted(self._enabled_ids)

    async def _get_writer_if_enabled(self, msg_id: str) -> tuple[bool, FileWriter | None, bool]:
        """
        Returns:
            enabled: whether msg_id is currently enabled
            writer: existing or newly-created writer if enabled
            need_open: whether writer.open_sync() still needs to be called
        """
        async with self._lock:
            if msg_id not in self._enabled_ids:
                return False, None, False

            writer = self._writers.get(msg_id)
            if writer is None:
                filename = self._filename_for_id(msg_id)
                writer = FileWriter(str(filename))
                self._writers[msg_id] = writer
                return True, writer, True

            return True, writer, False

    async def _disable_id_after_write_error(self, msg_id: str) -> None:
        writer = None
        async with self._lock:
            self._enabled_ids.discard(msg_id)
            writer = self._writers.pop(msg_id, None)

        if writer is not None:
            try:
                await asyncio.to_thread(writer.close_sync)
            except Exception as e:
                print(f"[QueueHandler] failed to close errored writer for {msg_id}: {e}")

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

                    enabled, writer, need_open = await self._get_writer_if_enabled(item.id)
                    if not enabled or writer is None:
                        continue

                    if need_open:
                        await asyncio.to_thread(writer.open_sync)

                    await asyncio.to_thread(writer.write_sync, item.message)

                    if writer.should_flush(self.flush_every):
                        await asyncio.to_thread(writer.flush_sync)

                    self.msg_count += 1
                    if self.msg_count % self.debug_every == 0:
                        self._debug_status()

                except OSError as e:
                    print(f"[QueueHandler] I/O error while writing {getattr(item, 'id', None)}: {e}")
                    if isinstance(item, QueueMessage):
                        await self._disable_id_after_write_error(item.id)

                except Exception as e:
                    print(f"[QueueHandler] unexpected error: {e}")

                finally:
                    self.queue.task_done()

        except asyncio.CancelledError:
            print("[QueueHandler] cancelled")
            raise

        finally:
            async with self._lock:
                writers = list(self._writers.values())
                self._writers.clear()

            for writer in writers:
                try:
                    await asyncio.to_thread(writer.close_sync)
                except Exception as e:
                    print(f"[QueueHandler] failed to close writer during shutdown: {e}")