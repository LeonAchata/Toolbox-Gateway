"""Pick a model from phrases like "use gemini" or "con gpt" in the prompt."""

from __future__ import annotations

import re

_VERB = r"(?:use|using|with|via|usa|usando|utiliza|con)"
_PATTERNS = [
    ("openai", rf"\b{_VERB}\s+(?:openai|gpt[\w.-]*|chatgpt)\b"),
    ("gemini", rf"\b{_VERB}\s+(?:gemini[\w.-]*|google)\b"),
    ("bedrock", rf"\b{_VERB}\s+(?:bedrock|nova|aws)\b"),
]


def detect_model(text: str) -> str | None:
    lowered = text.lower()
    matches = []
    for model, pattern in _PATTERNS:
        found = re.search(pattern, lowered)
        if found:
            matches.append((found.start(), model))
    # If several are mentioned, the first one wins.
    return min(matches)[1] if matches else None
