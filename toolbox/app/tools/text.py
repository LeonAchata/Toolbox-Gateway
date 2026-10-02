"""Text manipulation tools."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from ..registry import registry

Text = Annotated[str, Field(description="Input text", max_length=100_000)]


@registry.register("text")
def uppercase(text: Text) -> str:
    """Convert text to uppercase."""
    return text.upper()


@registry.register("text")
def lowercase(text: Text) -> str:
    """Convert text to lowercase."""
    return text.lower()


@registry.register("text")
def reverse_text(text: Text) -> str:
    """Reverse the order of the characters in a text."""
    return text[::-1]


@registry.register("text")
def count_words(text: Text) -> int:
    """Count the words in a text, splitting on whitespace."""
    return len(text.split())


@registry.register("text")
def count_characters(
    text: Text,
    include_spaces: Annotated[bool, Field(description="Whether whitespace counts")] = True,
) -> int:
    """Count the characters in a text, optionally ignoring whitespace."""
    if include_spaces:
        return len(text)
    return sum(1 for char in text if not char.isspace())
