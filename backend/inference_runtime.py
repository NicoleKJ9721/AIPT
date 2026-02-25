from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


InferenceKind = Literal["model", "pipeline"]
InferenceFormat = Literal["tensorrt", "openvino"]


@dataclass
class InferenceSessionRuntime:
    id: str
    key: str
    kind: InferenceKind
    target_id: str
    target_name: str
    project_id: str
    format: InferenceFormat | None
    device: str | None
    end2end: bool
    artifact_path: Path | None
    model: object | None
    created_at: str
    last_used_at: str | None = None
    pipeline_steps_snapshot: list[dict] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_requests: int = 0


class InferenceSessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, InferenceSessionRuntime] = {}
        self._session_by_key: dict[str, str] = {}
        self._active_requests = 0
        self._loading: dict[str, threading.Event] = {}
        self._loading_errors: dict[str, str] = {}

    def block_reason(self) -> str | None:
        with self._lock:
            if self._loading:
                return "Inference session is initializing"
            if self._active_requests > 0:
                return f"Inference is running (active_requests={self._active_requests})"
            if self._sessions:
                any_id = next(iter(self._sessions.keys()))
                return f"Inference session is active (session_id={any_id})"
        return None

    def snapshot(self) -> tuple[int, list[InferenceSessionRuntime]]:
        with self._lock:
            active = int(self._active_requests)
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return active, sessions

    def reserve_or_get(self, key: str, wait_timeout: int = 1200) -> tuple[str, InferenceSessionRuntime | None]:
        with self._lock:
            existing_id = self._session_by_key.get(key)
            if existing_id and existing_id in self._sessions:
                return "existing", self._sessions[existing_id]

            evt = self._loading.get(key)
            if evt is None:
                evt = threading.Event()
                self._loading[key] = evt
                return "loader", None

        evt.wait(timeout=wait_timeout)
        with self._lock:
            existing_id = self._session_by_key.get(key)
            if existing_id and existing_id in self._sessions:
                return "existing", self._sessions[existing_id]
            err = self._loading_errors.pop(key, None)
        raise RuntimeError(err or "Inference session initialization failed")

    def register_loaded(self, key: str, session: InferenceSessionRuntime) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._session_by_key[key] = session.id

    def finish_loading(self, key: str, error: str | None = None) -> None:
        evt = None
        with self._lock:
            if error:
                self._loading_errors[key] = str(error)
            evt = self._loading.pop(key, None)
        if evt is not None:
            evt.set()

    def get(self, session_id: str) -> InferenceSessionRuntime | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> InferenceSessionRuntime:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Inference session not found")
            if session.active_requests > 0:
                raise RuntimeError("Inference session is busy")
            self._sessions.pop(session_id, None)
            self._session_by_key.pop(session.key, None)
            return session

    def begin_request(self, session_id: str) -> InferenceSessionRuntime:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Inference session not found")
            self._active_requests += 1
            session.active_requests += 1
            return session

    def end_request(self, session_id: str) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            session = self._sessions.get(session_id)
            if session is not None:
                session.active_requests = max(0, session.active_requests - 1)

    def touch(self, session_id: str, when: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used_at = when
