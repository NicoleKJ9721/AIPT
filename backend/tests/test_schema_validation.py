import os
import sys

import pytest
from pydantic import ValidationError


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import AnnotationCreate  # noqa: E402


def test_annotation_create_rect_missing_fields():
    with pytest.raises(ValidationError):
        AnnotationCreate(type="rect", label="hd", x=1, y=2, width=10)


def test_annotation_create_polygon_invalid_points():
    with pytest.raises(ValidationError):
        AnnotationCreate(type="polygon", label="hd", x=0, y=0, points=[0, 1, 2, 3, 4])

