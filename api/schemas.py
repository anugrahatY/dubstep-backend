"""Request and response contracts for the public API."""

from typing import Literal

from pydantic import BaseModel, Field, root_validator


class AdaptRequest(BaseModel):
    # Japanese source line. `ja` is retained to match the model's existing API.
    ja: str = Field(min_length=1, max_length=2000, description="Japanese speech/dialogue to adapt")
    en: str | None = Field(default=None, max_length=2000, description="Optional English meaning/pivot")
    context: str = Field(default="", max_length=4000,
                         description="Scene, speaker, relationship, tone, or surrounding dialogue")
    slot_seconds: float | None = Field(default=None, gt=0, le=60)
    target_syllables: int | None = Field(default=None, ge=1, le=300)
    styles: list[Literal["hindi", "hinglish"]] = Field(default_factory=lambda: ["hindi", "hinglish"])
    limit: int = Field(default=10, ge=1, le=10, description="Maximum number of options to return")

    @root_validator(skip_on_failure=True)
    def require_timing_hint(cls, values):
        if values.get("slot_seconds") is not None and values.get("target_syllables") is not None:
            raise ValueError("Provide slot_seconds or target_syllables, not both")
        if not values.get("styles"):
            raise ValueError("Select at least one output style")
        return values


class AdaptOption(BaseModel):
    text: str
    style: Literal["hindi", "hinglish"]
    syllables: int
    target_syllables: int
    timing_error_pct: float
    meaning_score: float
    fit_score: float
    score: float
    flags: list[str] = Field(default_factory=list)


class AdaptResponse(BaseModel):
    english_pivot: str
    target_syllables: int
    options: list[AdaptOption]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class PhraseTranslateRequest(BaseModel):
    ja: str = Field(min_length=1, max_length=2000, description="Japanese dialogue, idiom, or phrase")
    context: str | dict[str, str] = Field(default="", description="Optional scene, speaker, or relationship context")
    provider: Literal["openai", "groq", "local"] = "groq"
    slot_seconds: float | None = Field(default=None, gt=0, le=60)
    limit: int = Field(default=10, ge=1, le=10)


class PhraseOption(BaseModel):
    text: str
    score: int = Field(ge=0, le=100)
    note: str


class PhraseTranslateResponse(BaseModel):
    options: list[PhraseOption]
