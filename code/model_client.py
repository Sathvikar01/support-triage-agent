from typing import Dict, List, Optional

from config import XIAOMI_API_KEY, XIAOMI_BASE_URL, XIAOMI_MODEL, XIAOMI_TIMEOUT_SECONDS


class ModelClient:
    """Thin OpenAI-compatible adapter with deterministic offline fallback."""

    def __init__(self):
        self.model = XIAOMI_MODEL
        self.base_url = XIAOMI_BASE_URL
        self.api_key = XIAOMI_API_KEY
        self.timeout = XIAOMI_TIMEOUT_SECONDS

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> Optional[str]:
        if not self.available:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=1,
            )
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"    [model API error: {type(exc).__name__}]")
            return None
