from __future__ import annotations
from bayyina.types import Hit

def _collapse(value: str) -> str:
    return " ".join(value.split())

def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    if not isinstance(quote, str) or not isinstance(document, str):
        return False
    if not (0 <= start < end <= len(document)):
        return False
    q = _collapse(quote)
    return bool(q) and q in _collapse(document[start:end])

def format_citation(hit: Hit, max_quote: int = 160) -> str:
    if max_quote < 0:
        raise ValueError("max_quote must be non-negative")
    quote = _collapse(hit.text)
    if len(quote) > max_quote:
        if max_quote == 0:
            quote = "…"
        else:
            prefix = quote[:max_quote]
            space = prefix.rfind(" ")
            if space >= 0:
                prefix = prefix[:space]
            quote = prefix + "…"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
