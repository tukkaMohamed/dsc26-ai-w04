from __future__ import annotations
from bayyina.types import Chunk

_SENTENCE = {".", "!", "?", "؟", "۔", "\n"}

def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("require max_chars > 0 and 0 <= overlap < max_chars")
    if not text or not text.strip():
        return []
    result = []
    start, n = 0, len(text)
    previous_midword_cut = False
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        if n - start <= max_chars:
            end = n
        else:
            window = text[start:start + max_chars]
            low = max_chars // 2
            paras = [i + 2 for i in range(low, max_chars - 1) if window[i:i+2] == "\n\n"]
            sentences = [i+1 for i,c in enumerate(window) if low <= i+1 <= max_chars and c in _SENTENCE]
            spaces = [i+1 for i,c in enumerate(window) if low <= i+1 <= max_chars and c.isspace()]
            cut = max(paras) if paras else max(sentences) if sentences else max(spaces) if spaces else max_chars
            end = start + cut
        while end > start and text[end-1].isspace():
            end -= 1
        if end <= start:
            start += 1
            continue
        idx = len(result)
        result.append(Chunk(doc_id, f"{doc_id}#{idx}", start, end, text[start:end]))
        if end >= n:
            break
        midword = end > 0 and end < n and not text[end-1].isspace() and not text[end].isspace()
        proposed = max(start + 1, end - overlap)
        if midword:
            # A hard/mid-word cut is the only case where the next chunk may
            # begin mid-word; retain the requested maximum overlap.
            previous_midword_cut = True
        else:
            previous_midword_cut = False
            # Choose the first word boundary at or after the overlap threshold.
            # Moving backwards could make the overlap larger than requested.
            while proposed < n and (text[proposed].isspace() or
                                    (proposed > 0 and not text[proposed-1].isspace())):
                proposed += 1
        if proposed <= start:
            proposed = end
        start = proposed
    return result
