"""Inference boundary. Replace or extend this service without changing routes."""

from typing import Protocol


class Adapter(Protocol):
    def adapt(self, ja, en=None, slot_seconds=None, target_syllables=None,
              n=10, styles=("hindi", "hinglish"), context="") -> dict: ...


class DubAdaptService:
    """Loads the model once and keeps the HTTP layer independent of inference."""

    def __init__(self, model_dir="model", use_labse=True, adapter: Adapter | None = None):
        if adapter is None:
            # Heavy ML dependencies are imported only when the app starts.
            from suggest import DubAdapter
            adapter = DubAdapter(model_dir, use_labse=use_labse)
        self.adapter = adapter

    def make_suggestions(self, request):
        result = self.adapter.adapt(
            ja=request.ja,
            en=request.en,
            slot_seconds=request.slot_seconds,
            target_syllables=request.target_syllables,
            n=request.limit,
            styles=tuple(request.styles),
            context=request.context,
        )
        # Enforce the API contract even if a future inference backend returns more.
        result["options"] = result.get("options", [])[:request.limit]
        return result
