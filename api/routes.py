"""Versioned HTTP routes."""

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    AdaptRequest,
    AdaptResponse,
    HealthResponse,
    PhraseTranslateRequest,
    PhraseTranslateResponse,
)
from .groq_translator import GroqTranslatorError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "adapter_service", None) is not None}


@router.post("/adapt", response_model=AdaptResponse)
def adapt(payload: AdaptRequest, request: Request):
    service = getattr(request.app.state, "adapter_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready")
    # Generation on one small GPU is serialized; a queue/worker can replace this
    # when deploying across multiple workers or GPU devices.
    with request.app.state.generation_lock:
        return service.make_suggestions(payload)


@router.post("/translate", response_model=PhraseTranslateResponse)
def translate(payload: PhraseTranslateRequest, request: Request):
    translators = getattr(request.app.state, "phrase_translators", {})
    translator = translators.get(payload.provider)
    if translator is None:
        if payload.provider == "openai":
            detail = "OpenAI translation is unavailable. Configure OPENAI_API_KEY in the backend environment."
        elif payload.provider == "local":
            detail = "The local model is unavailable. Configure MODEL_DIR and install the model requirements."
        else:
            detail = "Groq translation service is not ready. Configure GROQ_API_KEY in the backend environment."
        raise HTTPException(status_code=503, detail=detail)
    try:
        return translator.translate(payload)
    except GroqTranslatorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
