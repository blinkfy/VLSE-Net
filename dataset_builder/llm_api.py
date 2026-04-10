from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv

from openai import OpenAI

# Load .env from workspace/project root (if present). This populates os.environ but does not overwrite already-set env vars.
load_dotenv()


def generate_description(
    prompt: str,
    model: str = "gpt-5.4-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    # Resolve api_key/base_url/model from arguments -> environment variables -> defaults
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model = model or os.getenv("OPENAI_MODEL") or "gpt-5.4-mini"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    attempts = 0
    last_error: Exception | None = None

    while attempts < max_retries:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
            )
            return {
                "text": response.choices[0].message.content.strip(),
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
