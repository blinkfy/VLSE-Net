from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a local .env file when present.
load_dotenv()


def generate_description(
    prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Generate one text description through an OpenAI-compatible API.

    The model must be supplied explicitly either through ``model`` or the
    ``OPENAI_MODEL`` environment variable. This avoids silently changing the
    language model used for prompt generation.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model = model or os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not model:
        raise RuntimeError(
            "No language model is specified. Pass --model or set OPENAI_MODEL. "
            "For reproducing the paper prompt-generation procedure, configure "
            "your OpenAI-compatible provider to use Qwen3.5-Plus."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    attempts = 0
    last_error: Exception | None = None

    while attempts < max_retries:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("The language model returned an empty response.")

            return {
                "text": content.strip(),
                "mode": "text",
                "attempts": attempts,
                "fallback_used": False,
            }
        except Exception as exc:
            last_error = exc
            if attempts >= max_retries:
                raise
            time.sleep(retry_delay * attempts)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected failure in generate_description")
