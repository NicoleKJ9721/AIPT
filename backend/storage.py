from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

from fastapi import UploadFile

from app_config import load_settings


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_EXTENDED_PATH_THRESHOLD = 240


def filesystem_path(path: Path, *, force_extended: bool = False) -> Path:
    """
    Return a path that is safe for local filesystem operations.

    Windows still rejects many normal paths beyond ``MAX_PATH``. Dataset files
    live under several UUID directories, so a perfectly valid user-selected
    storage root can otherwise make uploads fail. Use the Win32 extended-path
    form only for long absolute paths; database values and API responses remain
    normal, portable relative paths.
    """
    expanded = path.expanduser()
    if os.name != "nt":
        return expanded

    raw = str(expanded)
    if raw.startswith("\\\\?\\"):
        return expanded

    if not expanded.is_absolute():
        raw = str(expanded.resolve())

    if not force_extended and len(raw) < _WINDOWS_EXTENDED_PATH_THRESHOLD:
        return Path(raw)

    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)


def legacy_storage_root() -> Path:
    """
    Legacy default root (inside the repo). Kept for backward compatibility with
    existing databases created before per-project storage roots were added.
    """
    base_dir = Path(__file__).resolve().parent
    return (base_dir / "data" / "storage").resolve()


def default_storage_root() -> Path:
    """
    Default storage root for newly created projects.
    - If `AIPT_STORAGE_DIR` is set, it wins (backward compatible).
    - Otherwise, use persisted system settings (outside repo by default).
    """
    env = os.getenv("AIPT_STORAGE_DIR")
    if env:
        return Path(env).expanduser().resolve()

    try:
        settings = load_settings()
        return Path(settings["projects_root_dir"]).expanduser().resolve()
    except Exception:
        # Fallback to a per-user dir (still outside repo).
        return (Path.home() / ".aipt" / "storage").resolve()


def project_storage_root(storage_root_value: str | None) -> Path:
    """
    Resolve the storage root that a given project should use.
    """
    if storage_root_value and storage_root_value.strip():
        return Path(storage_root_value).expanduser().resolve()
    return legacy_storage_root()


def storage_root() -> Path:
    """
    Backward-compatible alias used by older callers.
    Prefer passing an explicit root per project.
    """
    return default_storage_root()


def project_dir(project_id: str, root: Path | None = None) -> Path:
    base = (root or storage_root()).expanduser().resolve()
    return base / "projects" / project_id


def ensure_project_dirs(project_id: str, root: Path | None = None) -> None:
    """
    Create the project-level storage layout.
    """
    base = root or storage_root()
    ensure_dir(project_dir(project_id, root=base) / "datasets")
    ensure_dir(project_dir(project_id, root=base) / "exports")
    ensure_dir(project_dir(project_id, root=base) / "models")


def dataset_dir(dataset_id: str, project_id: str | None = None, root: Path | None = None) -> Path:
    base = root or storage_root()
    if project_id:
        return project_dir(project_id, root=base) / "datasets" / dataset_id
    return base / "datasets" / dataset_id


def ensure_dir(path: Path) -> None:
    filesystem_path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str, max_len: int = 200) -> str:
    name = filename.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_NAME_RE.sub("_", name)
    if not name:
        return "file"
    return name[:max_len]


def save_upload_file(
    upload: UploadFile,
    dataset_id: str,
    file_id: str,
    project_id: str | None = None,
    root: Path | None = None,
) -> tuple[str, int, str]:
    ensure_dir(dataset_dir(dataset_id, project_id=project_id, root=root))
    original = upload.filename or "file"
    stored_name = f"{file_id}__{safe_filename(original)}"
    dest = dataset_dir(dataset_id, project_id=project_id, root=root) / stored_name

    sha = hashlib.sha256()
    size = 0
    with filesystem_path(dest).open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            sha.update(chunk)
            size += len(chunk)

    if project_id:
        rel = Path("projects") / project_id / "datasets" / dataset_id / stored_name
    else:
        rel = Path("datasets") / dataset_id / stored_name
    return rel.as_posix(), size, sha.hexdigest()


def save_bytes(
    data: bytes,
    dataset_id: str,
    file_id: str,
    filename: str,
    project_id: str | None = None,
    root: Path | None = None,
) -> tuple[str, int, str]:
    ensure_dir(dataset_dir(dataset_id, project_id=project_id, root=root))
    stored_name = f"{file_id}__{safe_filename(filename)}"
    dest = dataset_dir(dataset_id, project_id=project_id, root=root) / stored_name

    sha = hashlib.sha256()
    sha.update(data)
    with filesystem_path(dest).open("wb") as f:
        f.write(data)

    if project_id:
        rel = Path("projects") / project_id / "datasets" / dataset_id / stored_name
    else:
        rel = Path("datasets") / dataset_id / stored_name
    return rel.as_posix(), len(data), sha.hexdigest()


def save_fileobj(
    fileobj: BinaryIO,
    dataset_id: str,
    file_id: str,
    filename: str,
    project_id: str | None = None,
    root: Path | None = None,
    chunk_size: int = 1024 * 1024,
) -> tuple[str, int, str]:
    """
    Stream a file-like object to disk and compute size + SHA256.

    This avoids loading entire ZIP members into memory when importing datasets.
    """
    ensure_dir(dataset_dir(dataset_id, project_id=project_id, root=root))
    stored_name = f"{file_id}__{safe_filename(filename)}"
    dest = dataset_dir(dataset_id, project_id=project_id, root=root) / stored_name

    sha = hashlib.sha256()
    size = 0
    with filesystem_path(dest).open("wb") as out:
        while True:
            chunk = fileobj.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            sha.update(chunk)
            size += len(chunk)

    if project_id:
        rel = Path("projects") / project_id / "datasets" / dataset_id / stored_name
    else:
        rel = Path("datasets") / dataset_id / stored_name
    return rel.as_posix(), size, sha.hexdigest()


def resolve_storage_path(rel_path: str, root: Path | None = None) -> Path:
    base = (root or storage_root()).resolve()
    target = (base / rel_path).resolve()
    if base not in target.parents and target != base:
        raise ValueError("Invalid storage path")
    return filesystem_path(target)


def delete_storage_path(rel_path: str, root: Path | None = None) -> None:
    path = resolve_storage_path(rel_path, root=root)
    if path.exists():
        path.unlink()


def zip_dataset_bytes(dataset_id: str, files: Iterable[tuple[str, str]], root: Path | None = None) -> bytes:
    """
    files: iterable of (display_filename, storage_rel_path)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for display_name, rel_path in files:
            path = resolve_storage_path(rel_path, root=root)
            if not path.exists():
                continue
            zf.write(path, arcname=f"{dataset_id}/{safe_filename(display_name)}")
    return buf.getvalue()
