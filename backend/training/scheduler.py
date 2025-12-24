from backend.models import TrainingJob

_JOBS: list[TrainingJob] = [
    TrainingJob(id="job-1", model="yolo-v8n", status="queued")
]


def list_jobs() -> list[TrainingJob]:
    return _JOBS
