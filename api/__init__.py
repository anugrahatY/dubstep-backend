"""HTTP API package for DubAdapt."""

from contextlib import asynccontextmanager
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .service import DubAdaptService
from .groq_translator import GroqPhraseTranslator, OpenAIPhraseTranslator, _backend_env_value
from .local_translator import LocalModelPhraseTranslator


def create_app(service=None, phrase_translator=None):
    """Build the app; allow a service override for alternate models/testing."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        project_root = Path(__file__).resolve().parents[2]
        modelbackend_dir = project_root / "modelbackend"
        if modelbackend_dir.is_dir() and str(modelbackend_dir) not in sys.path:
            sys.path.insert(0, str(modelbackend_dir))

        configured_model_dir = _backend_env_value("MODEL_DIR")
        model_dir = Path(configured_model_dir) if configured_model_dir else None
        if model_dir is not None and not model_dir.is_absolute():
            model_dir = project_root / model_dir
        adapter_service = service
        if adapter_service is None and model_dir and (model_dir / "adapter_config.json").is_file():
            try:
                adapter_service = DubAdaptService(
                    model_dir=str(model_dir),
                    use_labse=os.getenv("USE_LABSE", "true").lower() == "true",
                )
            except Exception as error:
                print(f"[Dubstep] Local model could not be loaded: {error}")
        app.state.adapter_service = adapter_service
        app.state.phrase_translators = {
            "groq": phrase_translator or GroqPhraseTranslator(),
            "openai": OpenAIPhraseTranslator(),
            "local": LocalModelPhraseTranslator(adapter_service) if adapter_service is not None else None,
        }
        app.state.generation_lock = threading.Lock()
        yield

    app = FastAPI(title="DubAdapt API", version="1.0.0", lifespan=lifespan)
    cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app
