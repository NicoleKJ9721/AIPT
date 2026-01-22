from __future__ import annotations

import datetime as dt
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int
    message: str
    data: T | None = None


class ProjectCreate(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    type: str = Field(default="目标检测", min_length=1, max_length=50)
    status: str = Field(default="进行中", min_length=1, max_length=50)
    latest_commit: str = Field(default="", max_length=200)
    storage_root: str | None = Field(default=None, max_length=2000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    type: str | None = Field(default=None, min_length=1, max_length=50)
    status: str | None = Field(default=None, min_length=1, max_length=50)
    latest_commit: str | None = Field(default=None, max_length=200)


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    type: str
    status: str
    latest_commit: str
    storage_root: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime
    images_count: int = 0


class ImageCreate(BaseModel):
    id: str | None = None
    filename: str = Field(min_length=1, max_length=300)
    source_url: str | None = Field(default=None, max_length=2000)
    width: int | None = Field(default=None, ge=1, le=100000)
    height: int | None = Field(default=None, ge=1, le=100000)


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    filename: str
    source_url: str | None
    dataset_id: str | None = None
    dataset_file_id: str | None = None
    width: int | None
    height: int | None
    created_at: dt.datetime
    annotations_count: int = 0


AnnotationType = Literal["rect", "polygon"]


class AnnotationCreate(BaseModel):
    id: str | None = None
    type: AnnotationType
    label: str = Field(min_length=1, max_length=200)
    color: str = Field(default="#ef4444", min_length=1, max_length=50)
    visible: bool = True

    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    points: list[float] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "AnnotationCreate":
        if self.type == "rect":
            missing = [k for k in ("x", "y", "width", "height") if getattr(self, k) is None]
            if missing:
                raise ValueError(f"rect annotation missing fields: {', '.join(missing)}")
        if self.type == "polygon":
            if not self.points or len(self.points) < 6 or len(self.points) % 2 != 0:
                raise ValueError("polygon annotation requires an even-length points list with >= 6 values")
        return self


class AnnotationUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=200)
    color: str | None = Field(default=None, min_length=1, max_length=50)
    visible: bool | None = None

    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    points: list[float] | None = None


class AnnotationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    image_id: str
    type: AnnotationType
    label: str
    color: str
    visible: bool
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    points: list[float] | None
    created_at: dt.datetime
    updated_at: dt.datetime


class LabelClassCreate(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    color: str = Field(default="#ef4444", min_length=1, max_length=50)
    shortcut: str | None = Field(default=None, max_length=20)


class LabelClassUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    color: str | None = Field(default=None, min_length=1, max_length=50)
    shortcut: str | None = Field(default=None, max_length=20)
    dataset_id: str | None = None


class LabelClassOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    name: str
    color: str
    shortcut: str
    created_at: dt.datetime
    updated_at: dt.datetime


class DatasetSplits(BaseModel):
    train: float = Field(default=0.7, ge=0, le=1)
    val: float = Field(default=0.2, ge=0, le=1)
    test: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_sum(self) -> "DatasetSplits":
        total = float(self.train) + float(self.val) + float(self.test)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("dataset splits must sum to 1.0")
        return self


class DatasetCreate(BaseModel):
    id: str | None = None
    project_id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    version: str | None = Field(default=None, min_length=1, max_length=50)
    description: str = Field(default="", max_length=2000)
    tags: list[str] | None = None
    is_public: bool = False
    splits: DatasetSplits = Field(default_factory=DatasetSplits)


class DatasetCloneRequest(BaseModel):
    id: str | None = None
    name: str | None = Field(default=None, min_length=1, max_length=200)
    version: str | None = Field(default=None, min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = None
    is_public: bool | None = None
    splits: DatasetSplits | None = None


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    version: str | None = Field(default=None, min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=2000)
    status: str | None = Field(default=None, min_length=1, max_length=50)
    tags: list[str] | None = None
    is_public: bool | None = None
    splits: DatasetSplits | None = None


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str | None
    name: str
    version: str
    description: str
    status: str
    owner: str
    is_public: bool
    tags: list[str] | None
    splits: DatasetSplits
    created_at: dt.datetime
    updated_at: dt.datetime
    file_count: int = 0
    total_size_bytes: int = 0


class DatasetFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: dt.datetime


class DatasetImageStatsOut(BaseModel):
    dataset_id: str
    image_count: int = 0
    sized_image_count: int = 0
    avg_width: int | None = None
    avg_height: int | None = None
    total_pixels: int = 0


class TrainedModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_id: str | None = None
    name: str
    base_model: str
    metrics: dict | None = None
    created_at: dt.datetime


class ModelEvaluationDetectionOut(BaseModel):
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int


class ModelEvaluationItemOut(BaseModel):
    image: ImageOut
    detections: list[ModelEvaluationDetectionOut] = Field(default_factory=list)


class ModelEvaluationPageOut(BaseModel):
    model_id: str
    dataset_id: str
    split: Literal["train", "val", "test"]
    page: int
    limit: int
    total: int
    note: str | None = None
    items: list[ModelEvaluationItemOut] = Field(default_factory=list)


class SystemSettingsOut(BaseModel):
    projects_root_dir: str
    resources_root_dir: str
    recent_projects_root_dirs: list[str] = Field(default_factory=list)
    recent_resources_root_dirs: list[str] = Field(default_factory=list)
    default_model_resource_id: str | None = None


class SystemSettingsUpdate(BaseModel):
    projects_root_dir: str | None = None
    resources_root_dir: str | None = None
    default_model_resource_id: str | None = None


class HardwareDeviceOut(BaseModel):
    id: str
    name: str
    type: str
    vendor: str | None = None
    memory: str | None = None
    cores: int | None = None
    compute_capability: str | None = None
    status: str = "Available"
