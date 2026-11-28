"""OCR by the LLM: send a picture of the page and ask for its text (OCR_PROVIDER=llm).

For hosts that can't run the Tesseract program (e.g. Vercel). Gemini can read images, so
scanned PDFs and photos still work. Each page costs one LLM request (Gemini's free tier
allows about 20 a day), so a long scan uses a lot of the daily quota.
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image

from app.core.config import Settings
from app.core.exceptions import ExtractionError, LLMNotConfiguredError, LLMRateLimitError

logger = logging.getLogger(__name__)

_MAX_SIDE = 2000  # pixels; enough to read normal text, and keeps each request small
_PROMPT = (
    "Transcribe all text in this image exactly as written, keeping the reading order and line "
    "breaks. Include text in tables. Reply with the text only, no comments. If there is no text, "
    "reply with nothing."
)


class LLMImageReader:
    def __init__(self, settings: Settings) -> None:
        import openai

        if not settings.llm_configured:
            raise LLMNotConfiguredError("OCR_PROVIDER=llm needs LLM_API_KEY (and LLM_BASE_URL) in .env.")
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.llm_model

    def __call__(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((_MAX_SIDE, _MAX_SIDE))  # shrink big scans, never enlarge
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

        openai = self._openai
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            )
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(
                "The AI service is rate-limited, so the scanned pages could not be read. Try again later."
            ) from exc
        except openai.APIError as exc:
            logger.error("LLM OCR failed: %s", type(exc).__name__)
            raise ExtractionError("The AI service could not read the scanned page.") from exc
        return (response.choices[0].message.content or "").strip() if response.choices else ""
