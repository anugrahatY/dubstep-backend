"""Adapter for the in-repository Qwen LoRA model backend."""

from __future__ import annotations

import threading

from .groq_translator import GroqTranslatorError


class LocalModelPhraseTranslator:
    def __init__(self, adapter_service):
        self.adapter_service = adapter_service
        self.generation_lock = threading.Lock()

    @staticmethod
    def _model_context(context) -> dict:
        if not isinstance(context, dict):
            context = {}
        return {
            "speaker": {"gender": context.get("speaker")},
            "listener": {
                "gender": context.get("listener"),
                "relation": context.get("relation"),
            },
            "scene": {"emotion": context.get("emotion")},
            "pronoun": context.get("pronoun"),
        }

    def translate(self, request) -> dict:
        if self.adapter_service is None:
            raise GroqTranslatorError(
                "The local model is not loaded. Install the model requirements and set MODEL_DIR to its adapter checkpoint.",
                status_code=503,
            )

        try:
            with self.generation_lock:
                result = self.adapter_service.adapter.adapt(
                    request.ja,
                    slot_seconds=request.slot_seconds,
                    n=request.limit,
                    styles=("hindi",),
                    context=self._model_context(request.context),
                )
        except Exception:
            raise GroqTranslatorError(
                "Local model inference failed. Check MODEL_DIR, model dependencies, and available memory.",
                status_code=500,
            ) from None

        options = []
        for option in result.get("options", [])[:request.limit]:
            score = max(0, min(100, round(float(option.get("score", 0)) * 100)))
            meaning = option.get("meaning_score")
            timing = option.get("timing_error_pct")
            details = []
            if meaning is not None:
                details.append(f"Meaning match {round(float(meaning) * 100)}%")
            if timing is not None:
                details.append(f"{float(timing):.1f}% timing difference")
            options.append({
                "text": option.get("text", ""),
                "score": score,
                "note": "; ".join(details) or "Ranked by the local dubbing model.",
            })
        return {"options": options}
