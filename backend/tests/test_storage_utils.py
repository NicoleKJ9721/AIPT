import io
import os
import sys
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage import (  # noqa: E402
    default_storage_root,
    legacy_storage_root,
    project_storage_root,
    resolve_storage_path,
    safe_filename,
    save_bytes,
    save_fileobj,
    save_upload_file,
    storage_root,
    zip_dataset_bytes,
    dataset_dir,
)


def test_safe_filename_sanitizes_paths():
    assert safe_filename("../a/b.png") == "b.png"
    assert safe_filename("..\\evil\\name?.jpg").startswith("name_")
    assert safe_filename(" ") == "file"


def test_resolve_storage_path_blocks_traversal(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_storage_path("../../evil.txt", root=tmp_path)


def test_save_fileobj_streams_and_hashes(tmp_path: Path):
    project_id = "p-" + uuid4().hex[:8]
    dataset_id = "d-" + uuid4().hex[:8]
    file_id = "f-" + uuid4().hex[:8]
    filename = "hello.txt"
    payload = b"hello world"

    storage_rel, size, digest = save_fileobj(
        io.BytesIO(payload),
        dataset_id=dataset_id,
        file_id=file_id,
        filename=filename,
        project_id=project_id,
        root=tmp_path,
        chunk_size=4,
    )

    assert size == len(payload)
    assert digest == sha256(payload).hexdigest()

    path = resolve_storage_path(storage_rel, root=tmp_path)
    assert path.exists()
    assert path.read_bytes() == payload


def test_save_fileobj_without_project_id_uses_dataset_root(tmp_path: Path):
    dataset_id = "d-" + uuid4().hex[:8]
    file_id = "f-" + uuid4().hex[:8]

    storage_rel, size, digest = save_fileobj(
        io.BytesIO(b"abc"),
        dataset_id=dataset_id,
        file_id=file_id,
        filename="a.txt",
        project_id=None,
        root=tmp_path,
        chunk_size=2,
    )

    assert storage_rel.startswith(f"datasets/{dataset_id}/")
    assert size == 3
    assert digest == sha256(b"abc").hexdigest()


def test_save_bytes_persists_and_hashes(tmp_path: Path):
    dataset_id = "d-" + uuid4().hex[:8]
    file_id = "f-" + uuid4().hex[:8]
    payload = b"payload"

    storage_rel, size, digest = save_bytes(
        payload,
        dataset_id=dataset_id,
        file_id=file_id,
        filename="hello.bin",
        project_id=None,
        root=tmp_path,
    )

    assert storage_rel.startswith(f"datasets/{dataset_id}/")
    assert size == len(payload)
    assert digest == sha256(payload).hexdigest()

    path = resolve_storage_path(storage_rel, root=tmp_path)
    assert path.exists()
    assert path.read_bytes() == payload


def test_save_bytes_with_project_id_uses_project_root(tmp_path: Path):
    project_id = "p-" + uuid4().hex[:8]
    dataset_id = "d-" + uuid4().hex[:8]
    file_id = "f-" + uuid4().hex[:8]

    storage_rel, size, digest = save_bytes(
        b"abc",
        dataset_id=dataset_id,
        file_id=file_id,
        filename="x.bin",
        project_id=project_id,
        root=tmp_path,
    )

    assert storage_rel.startswith(f"projects/{project_id}/datasets/{dataset_id}/")
    assert size == 3
    assert digest == sha256(b"abc").hexdigest()

def test_save_upload_file_supports_non_project_datasets(tmp_path: Path):
    class DummyUpload:
        def __init__(self, filename: str, data: bytes):
            self.filename = filename
            self.file = io.BytesIO(data)

    dataset_id = "d-" + uuid4().hex[:8]
    file_id = "f-" + uuid4().hex[:8]
    payload = b"hello"

    storage_rel, size, digest = save_upload_file(
        DummyUpload("x.txt", payload),
        dataset_id=dataset_id,
        file_id=file_id,
        project_id=None,
        root=tmp_path,
    )

    assert storage_rel.startswith(f"datasets/{dataset_id}/")
    assert size == len(payload)
    assert digest == sha256(payload).hexdigest()


def test_default_storage_root_prefers_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "env_root"))
    assert default_storage_root() == (tmp_path / "env_root").resolve()


def test_default_storage_root_uses_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("AIPT_STORAGE_DIR", raising=False)
    monkeypatch.setattr(
        sys.modules["storage"],
        "load_settings",
        lambda: {"projects_root_dir": str(tmp_path / "settings_root")},
    )
    assert default_storage_root() == (tmp_path / "settings_root").resolve()


def test_default_storage_root_falls_back_on_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AIPT_STORAGE_DIR", raising=False)
    monkeypatch.setattr(sys.modules["storage"], "load_settings", lambda: (_ for _ in ()).throw(RuntimeError("bad")))

    # Ensure it returns a deterministic location even when settings loading fails.
    assert default_storage_root().name == "storage"


def test_project_storage_root_and_dataset_dir_fallbacks(tmp_path: Path):
    assert project_storage_root(None) == legacy_storage_root()
    assert dataset_dir("d1", project_id=None, root=tmp_path) == (tmp_path / "datasets" / "d1")


def test_storage_root_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "env_root"))
    assert storage_root() == (tmp_path / "env_root").resolve()


def test_zip_dataset_bytes_skips_missing_files(tmp_path: Path):
    payload = zip_dataset_bytes("ds1", files=[("missing.txt", "datasets/ds1/missing.txt")], root=tmp_path)
    assert payload  # valid zip bytes even if empty
