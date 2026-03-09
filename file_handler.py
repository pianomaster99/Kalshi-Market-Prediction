import gzip
from typing import Iterator, Optional, TextIO


class FileWriter:
    def __init__(self, filename: str):
        self.filename = filename
        self._file: Optional[TextIO] = None

    def open(self) -> None:
        if self._file is None:
            self._file = gzip.open(self.filename, "at", encoding="utf-8")

    def write(self, message: str) -> None:
        if self._file is None:
            raise RuntimeError(f"Writer for {self.filename} is not open")
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

def file_read(path: str) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line