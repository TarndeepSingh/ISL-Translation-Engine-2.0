"""Versioned, task-separated inference API. Model absence is explicit."""
from __future__ import annotations

import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .media import UploadProblem, discard_upload, run_bounded_inference, stage_upload
from .schemas import ReadinessResponse, RecognitionResponse, TranslationResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["inference-v1"])


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _model_version(pipeline) -> str:
    return str(getattr(pipeline.predictor, "model_version", "legacy-unversioned"))


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/readiness", response_model=ReadinessResponse)
def readiness(request: Request):
    isolated = getattr(request.app.state, "pipeline", None)
    continuous = getattr(request.app.state, "translation_pipeline", None)
    iso_status = {"available": isolated is not None,
                  "model_version": _model_version(isolated) if isolated else None,
                  "reason_code": None if isolated else "MODEL_NOT_READY",
                  "reason": None if isolated else "No isolated recognition model package is loaded."}
    cont_status = {"available": continuous is not None,
                   "model_version": getattr(continuous, "model_version", None) if continuous else None,
                   "reason_code": None if continuous else "MODEL_NOT_READY",
                   "reason": None if continuous else "No continuous translation model package is loaded."}
    count = int(iso_status["available"]) + int(cont_status["available"])
    status = "ready" if count == 2 else "partially_ready" if count else "not_ready"
    return {"status": status, "isolated_recognition": iso_status, "continuous_translation": cont_status}


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize(request: Request, file: UploadFile | None = File(default=None)):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise _error(503, "MODEL_NOT_READY", "Isolated recognition is unavailable until a model package is loaded.")
    if file is None:
        raise _error(422, "VIDEO_REQUIRED", "Upload a video file for isolated recognition.")
    staged = None
    try:
        staged = await stage_upload(file)
        result = await run_bounded_inference(pipeline.predict_video, str(staged["path"]))
        top = [{"label": str(label), "score": float(score)} for label, score in result.get("top", [])]
        return {
            "task": "isolated", "status": str(result.get("status", "uncertain")),
            "prediction": str(result["label"]), "score": float(result["conf"]),
            "score_semantics": result.get("score_semantics", "raw_softmax_uncalibrated"),
            "margin": float(result["margin"]), "top_k": top,
            "quality": {str(k): float(v) for k, v in result.get("quality", {}).items()},
            "duration_seconds": float(result.get("duration_s", staged["media"]["duration_s"])),
            "processing_ms": float(result.get("ms", 0.0)),
            "model_version": _model_version(pipeline), "input_sha256": staged["sha256"],
        }
    except UploadProblem as exc:
        raise _error(exc.status_code, exc.code, exc.message) from exc
    except Exception as exc:
        log.exception("isolated inference failed")
        raise _error(422, "INFERENCE_FAILED", "The video could not be processed by the isolated model.") from exc
    finally:
        if staged:
            discard_upload(staged["path"])


@router.post("/translate", response_model=TranslationResponse)
async def translate(request: Request, file: UploadFile | None = File(default=None)):
    pipeline = getattr(request.app.state, "translation_pipeline", None)
    if pipeline is None:
        raise _error(503, "MODEL_NOT_READY", "Continuous translation is unavailable until its data and model are ready.")
    if file is None:
        raise _error(422, "VIDEO_REQUIRED", "Upload a video file for continuous translation.")
    staged = None
    try:
        staged = await stage_upload(file)
        result = await run_bounded_inference(pipeline.translate_video, str(staged["path"]))
        return {
            "task": "continuous", "status": "generated", "generated_text": str(result["text"]),
            "model_version": str(getattr(pipeline, "model_version", "unknown")),
            "processing_ms": float(result["ms"]), "input_sha256": staged["sha256"],
        }
    except UploadProblem as exc:
        raise _error(exc.status_code, exc.code, exc.message) from exc
    except Exception as exc:
        log.exception("continuous translation failed")
        raise _error(422, "INFERENCE_FAILED", "The video could not be processed by the translation model.") from exc
    finally:
        if staged:
            discard_upload(staged["path"])
