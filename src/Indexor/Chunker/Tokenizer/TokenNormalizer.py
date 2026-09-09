"""Shared token normalization used by indexing and search."""

import re
from typing import List


_CAMEL_PARTS = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+"
)


def normalize_identifier(identifier: str) -> List[str]:
    """Return a normalized identifier and its component tokens."""
    value = identifier.strip().lower()
    if not value:
        return []

    result = [value]
    for part in re.split(r"[_\-.]+", identifier):
        if not part:
            continue
        result.append(part.lower())
        result.extend(piece.lower() for piece in _CAMEL_PARTS.findall(part))

    unique: List[str] = []
    for token in result:
        if token and token not in unique:
            unique.append(token)
    return unique


def tokenize_text(source: str) -> List[str]:
    """Tokenize text using the normalization shared by indexing and queries."""
    result: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9_\-.]+", source):
        result.extend(normalize_identifier(raw))
    return result


def tokenize_range(source: str, start: int, end: int) -> List[str]:
    """Tokenize a source range while retaining tokens crossing its edges."""
    result: List[str] = []
    for match in re.finditer(r"[A-Za-z0-9_\-.]+", source):
        if match.start() >= end:
            break
        if match.end() > start:
            result.extend(normalize_identifier(match.group()))
    return result


def add_context_tokens(
        tokens: List[str], filename: str, kind: str,
        language: str = "python") -> List[str]:
    """Add filename and source-kind tokens to a chunk token list."""
    result = list(tokens)
    context = [*tokenize_text(filename), language, "code"]
    if kind == "class":
        context.append("class")
    elif kind == "function":
        context.append("function")
    for token in context:
        if token not in result:
            result.append(token)
    return result
