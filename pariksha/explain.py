"""Plain-language explanations of findings from the BharatCode API (OpenAI-compatible)."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .findings import Finding
from .rules import RULES

DEFAULT_BASE_URL = "https://bharatcode.ai/api/model/v1"
DEFAULT_MODEL = "qwen-3.8-27b"

SYSTEM_PROMPT = (
    "You review machine-learning code for evaluation mistakes. You are given one finding from a "
    "static checker and the code around it. In at most five short sentences, explain in plain "
    "words why this makes the reported results untrustworthy, then show the smallest corrected "
    "version of the marked lines as a Python code block. If the code shows the finding is a false "
    "alarm, say so and explain why. Do not invent numbers about the project."
)


class ExplainError(Exception):
    pass


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into the environment without overriding variables already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_prompt(finding: Finding) -> str:
    rule = RULES[finding.rule]
    return (
        f"Rule {finding.rule}: {rule.title}\n"
        f"Location: {finding.where}\n"
        f"Finding: {finding.message}\n"
        f"Standard fix: {rule.fix}\n\n"
        f"Code (the marked line is the finding):\n{finding.snippet()}\n"
    )


def explain(finding: Finding, *, api_key: str, base_url: str = DEFAULT_BASE_URL,
            model: str = DEFAULT_MODEL, timeout: float = 90) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(finding)},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as err:
        detail = err.read(300).decode("utf-8", "replace")
        raise ExplainError(f"BharatCode API returned HTTP {err.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as err:
        raise ExplainError(f"could not reach the BharatCode API: {err}") from None
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ExplainError("unexpected response from the BharatCode API") from None
    # Reasoning models may prepend their thinking; keep only the answer.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def settings() -> tuple[str | None, str, str]:
    load_dotenv()
    return (os.environ.get("BHARATCODE_API_KEY"),
            os.environ.get("BHARATCODE_BASE_URL", DEFAULT_BASE_URL),
            os.environ.get("BHARATCODE_MODEL", DEFAULT_MODEL))
