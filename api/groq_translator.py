"""Groq-backed Japanese phrase adaptation to natural, idiomatic Hindi."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GroqTranslatorError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _context_to_prompt(context) -> str:
    if isinstance(context, str):
        return context.strip() or "No extra scene context was provided."
    if not isinstance(context, dict):
        return "No extra scene context was provided."
    labels = (("speaker", "Speaker"), ("listener", "Listener"), ("relation", "Relationship"),
              ("pronoun", "Pronoun"), ("emotion", "Emotion"))
    lines = [f"{label}: {context[key]}" for key, label in labels if context.get(key)]
    return "\n".join(lines) or "No extra scene context was provided."


def _backend_env_value(name: str) -> str | None:
    """Read one value from backend/.env without exposing it or overriding env vars."""
    value = os.getenv(name)
    if value:
        return value

    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        key, separator, raw_value = candidate.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not separator or key != name:
            continue
        raw_value = raw_value.strip()
        if raw_value.startswith(("\"", "'")):
            quote = raw_value[0]
            closing_quote = raw_value.find(quote, 1)
            raw_value = raw_value[1:closing_quote] if closing_quote > 0 else raw_value[1:]
        else:
            raw_value = raw_value.split(" #", 1)[0].strip()
        return raw_value or None
    return None


class ChatCompletionPhraseTranslator:
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    key_env = "GROQ_API_KEY"
    model_env = "GROQ_MODEL"
    default_model = "openai/gpt-oss-120b"
    provider_name = "Groq"
    user_agent = "dubstep-backend/1.0 (Groq API client)"
    reasoning_effort = "medium"

    def __init__(self, *, api_key: str | None = None, model: str | None = None, timeout: float = 60):
        self.api_key = api_key
        self.model = model or _backend_env_value(self.model_env) or self.default_model
        self.timeout = timeout

    def _key(self) -> str:
        key = self.api_key or _backend_env_value(self.key_env)
        if not key:
            raise GroqTranslatorError(
                f"{self.key_env} is missing. Add it to backend/.env or the server environment.",
                status_code=503,
            )
        return key

    def translate(self, request) -> dict:
        limit = request.limit
        timing = (
            f"Keep each option concise enough to fit a spoken delivery of about {request.slot_seconds:g} seconds."
            if request.slot_seconds
            else "Keep the Hindi natural and concise for spoken dialogue."
        )
        context = _context_to_prompt(request.context)
        user_prompt = (
            f"Japanese line: {request.ja}\n"
            f"Scene and speaker context: {context}\n"
            f"Timing: {timing}\n\n"
            f"Return exactly {limit} distinct Hindi adaptations, ranked from best to least suitable. "
            "Each note should briefly describe the tone or nuance in English. Return a JSON object in exactly "
            'this shape: {"options":[{"text":"Hindi phrase in Devanagari","score":95,"note":"Brief nuance"}]}. '
            "Use only the options key; each score must be an integer from 0 to 100, and each option must contain "
            "only text, score, and note. Do not include markdown or surrounding commentary."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Japanese-to-Hindi dialogue translator. Identify the Japanese line's "
                        "intended meaning, emotion, and social function, then express it with the closest commonly "
                        "used Hindi idiom, proverb, or conversational phrase. Prefer an established Hindi expression "
                        "with the same meaning over a literal translation. Do not force an idiom if none is a close "
                        "match; use the most natural spoken Hindi phrase instead. Use the provided context to preserve "
                        "politeness, relationship, and tone. Write Hindi in Devanagari, not transliteration or Hinglish. "
                        f"Return {limit} distinct candidates, ranked by semantic and conversational closeness, with the "
                        "closest match first. Give each an integer match score from 0 to 100 and a brief English note. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 1,
            "max_completion_tokens": 4096,
            "top_p": 1,
            "stream": True,
            "stop": None,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        http_request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                content_parts = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices", [])
                    if choices:
                        text = choices[0].get("delta", {}).get("content")
                        if text:
                            content_parts.append(text)
                completion = {"choices": [{"message": {"content": "".join(content_parts)}}]}
        except HTTPError as error:
            raw_error = ""
            try:
                raw_error = error.read().decode("utf-8")
                provider_error = json.loads(raw_error)
                nested_error = provider_error.get("error", {})
                if isinstance(nested_error, dict):
                    detail = nested_error.get("message", "")
                elif isinstance(nested_error, str):
                    detail = nested_error
                else:
                    detail = ""
                if not detail and isinstance(provider_error, dict):
                    detail = provider_error.get("detail") or provider_error.get("message") or ""
            except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
                detail = raw_error
            except TypeError:
                detail = ""
            # The provider's message is useful for diagnosing model/account
            # access, but never reflect a credential if one appears in it.
            key = self.api_key or _backend_env_value(self.key_env) or ""
            if key and detail:
                detail = detail.replace(key, "[redacted]")
            detail = detail[:300].strip()
            if error.code == 401:
                raise GroqTranslatorError(
                    f"{self.provider_name} rejected the API key. Check {self.key_env}."
                ) from None
            if error.code == 429:
                raise GroqTranslatorError(
                    f"{self.provider_name} is rate-limiting requests. Please try again shortly.", status_code=429
                ) from None
            if error.code == 403 and not detail:
                detail = f"Check access to model {self.model} and your {self.provider_name} project settings."
            message = f"{self.provider_name} request failed with status {error.code}."
            if detail:
                message = f"{message} {detail}"
            # Preserve provider client errors (400-499) so callers see the
            # actual cause instead of a misleading backend 502.
            status_code = error.code if 400 <= error.code < 500 else 502
            raise GroqTranslatorError(message, status_code=status_code) from None
        except (URLError, TimeoutError):
            raise GroqTranslatorError(f"Could not reach {self.provider_name}. Please try again shortly.") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise GroqTranslatorError(f"{self.provider_name} returned an unreadable response.") from None

        try:
            content = completion["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise GroqTranslatorError(
                f"{self.provider_name} did not return valid JSON. Please retry; if it repeats, check the model response format."
            ) from None

        options = parsed.get("options") if isinstance(parsed, dict) else None
        if not isinstance(options, list):
            raise GroqTranslatorError(f"{self.provider_name} JSON response is missing the options array.")
        if len(options) != limit:
            raise GroqTranslatorError(
                f"{self.provider_name} returned {len(options)} options; expected {limit}. Please retry."
            )
        for index, item in enumerate(options, start=1):
            if not isinstance(item, dict):
                raise GroqTranslatorError(f"{self.provider_name} option {index} was not an object.")
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise GroqTranslatorError(f"{self.provider_name} option {index} is missing its Hindi text.")
            if not isinstance(item.get("score"), int) or not 0 <= item["score"] <= 100:
                raise GroqTranslatorError(f"{self.provider_name} option {index} has an invalid score.")
            if not isinstance(item.get("note"), str):
                raise GroqTranslatorError(f"{self.provider_name} option {index} is missing its note.")

        # Keep UI order tied to the actual match ranking even if the model orders
        # an occasional response inconsistently.
        options.sort(key=lambda option: option["score"], reverse=True)
        return {"options": options}


class GroqPhraseTranslator(ChatCompletionPhraseTranslator):
    pass


class OpenAIPhraseTranslator(ChatCompletionPhraseTranslator):
    endpoint = "https://api.openai.com/v1/chat/completions"
    key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    default_model = "gpt-6-luna"
    provider_name = "OpenAI"
    user_agent = "dubstep-backend/1.0 (OpenAI API client)"
    reasoning_effort = "none"
