"""Typed v1 API responses; isolated and continuous results remain distinct."""
from typing import Literal

from pydantic import BaseModel, Field


class TaskAvailability(BaseModel):
    available: bool
    model_version: str | None = None
    reason_code: str | None = None
    reason: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "partially_ready", "not_ready"]
    isolated_recognition: TaskAvailability
    continuous_translation: TaskAvailability


class TopPrediction(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class RecognitionResponse(BaseModel):
    task: Literal["isolated"] = "isolated"
    status: str
    prediction: str
    score: float = Field(ge=0.0, le=1.0)
    score_semantics: Literal["raw_softmax_uncalibrated", "temperature_scaled_validation_calibrated"] = "raw_softmax_uncalibrated"
    margin: float
    top_k: list[TopPrediction]
    quality: dict[str, float]
    duration_seconds: float
    processing_ms: float
    model_version: str
    input_sha256: str


class TranslationResponse(BaseModel):
    task: Literal["continuous"] = "continuous"
    generated_text: str
    model_version: str
    processing_ms: float
    input_sha256: str
    status: Literal["generated"] = "generated"
