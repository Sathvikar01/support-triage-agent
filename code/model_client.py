import logging
from typing import Dict, List, Optional

from config import XIAOMI_API_KEY, XIAOMI_BASE_URL, XIAOMI_MODEL, XIAOMI_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class ModelClient:
    """Thin OpenAI-compatible adapter with deterministic offline fallback."""

    def __init__(self):
        self.model = XIAOMI_MODEL
        self.base_url = XIAOMI_BASE_URL
        self.api_key = XIAOMI_API_KEY
        self.timeout = XIAOMI_TIMEOUT_SECONDS
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=1,
            )
        return self._client

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> Optional[str]:
        if not self.available:
            return None

        try:
            client = self._get_client()
        except ImportError:
            logger.warning("openai package not installed; falling back to template response.")
            return None

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Model API error: %s: %s", type(exc).__name__, exc)
            return None
