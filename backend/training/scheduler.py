from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingJob:
    id: str
    model: str
    status: str
    created_at: datetime


_JOBS: list[TrainingJob] = []


def list_jobs() -> list[TrainingJob]:
    return _JOBS
