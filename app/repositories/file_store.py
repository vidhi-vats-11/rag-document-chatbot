"""Where the original uploaded files are kept (the only module that touches data/uploads).

Locally they are plain files in UPLOAD_DIR. With a database (DATABASE_URL) they are kept in
Postgres instead (postgres.PgFileStore), because a host like Vercel has no shared disk.
Both have the same three methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.exceptions import AppError


class FileStore(Protocol):
    def save(self, name: str, content: bytes) -> None: ...
    def read(self, name: str) -> bytes | None: ...
    def delete(self, name: str) -> None: ...


class LocalFileStore:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir.resolve()

    def save(self, name: str, content: bytes) -> None:
        self._safe_path(name).write_bytes(content)

    def read(self, name: str) -> bytes | None:
        path = self._safe_path(name)
        return path.read_bytes() if path.exists() else None

    def delete(self, name: str) -> None:
        self._safe_path(name).unlink(missing_ok=True)

    def _safe_path(self, name: str) -> Path:
        """Resolve a path and refuse anything that escapes the upload directory."""
        path = (self.upload_dir / name).resolve()
        if not path.is_relative_to(self.upload_dir):
            raise AppError("Invalid storage path.")
        return path
