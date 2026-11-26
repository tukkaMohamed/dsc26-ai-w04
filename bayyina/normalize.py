from __future__ import annotations
import re

_DIACRITICS = {chr(c) for c in range(0x064B, 0x0653)} | {"\u0670"}
_ALEF = str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ٱ":"ا","ى":"ي","ة":"ه"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = "".join(c for c in text if c not in _DIACRITICS and c != "\u0640")
    text = text.translate(_ALEF).translate(_DIGITS).lower()
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_ar(text), flags=re.UNICODE)
