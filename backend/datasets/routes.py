from fastapi import APIRouter

from backend.models import Dataset
from backend.services.dataset_import import list_datasets

router = APIRouter()


@router.get("/", response_model=list[Dataset])
def get_datasets() -> list[Dataset]:
    return list_datasets()
