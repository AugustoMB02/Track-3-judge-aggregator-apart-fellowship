"""Minimal script to test Martian API via OpenAI SDK.

Per https://docs.withmartian.com/integrations/openai-sdk:
  - base_url: https://api.withmartian.com/v1
  - model: provider/model-name (e.g., openai/gpt-4.1-nano)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv


def _render_preview(payload: Any, *, limit: int = 200) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    load_dotenv()

    martian_key = os.environ.get("MARTIAN_API_KEY")
    martian_url = os.environ.get("MARTIAN_API_URL")
    if not martian_key or not martian_url:
        print("Missing MARTIAN_API_KEY or MARTIAN_API_URL environment variables.", file=sys.stderr)
        return 2

    model = os.environ.get("MARTIAN_MODEL", "openai/gpt-4.1-nano").strip()
    client = OpenAI(api_key=martian_key, base_url=martian_url.rstrip("/") + "/v1")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a simple connectivity check."},
                {"role": "user", "content": "Respond with pong."},
            ],
            max_tokens=16,
        )
    except OpenAIError as exc:
        detail = getattr(exc, "response", None)
        status = getattr(detail, "status_code", "?") if detail is not None else "?"
        if detail is not None:
            try:
                body = detail.json()
            except Exception:
                body = getattr(detail, "text", repr(detail))
        else:
            body = str(exc)
        print(f"Chat completion failed (status {status}): {_render_preview(body)}", file=sys.stderr)
        return 4

    print("Martian API response:")
    print(_render_preview(response.model_dump()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
