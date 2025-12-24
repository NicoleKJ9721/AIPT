from fastapi import APIRouter

from backend.training.scheduler import TrainingJob, list_jobs

router = APIRouter()


@router.get("/jobs", response_model=list[TrainingJob])
def get_jobs() -> list[TrainingJob]:
    return list_jobs()
