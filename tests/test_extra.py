import pytest
from bayyina.normalize import normalize_ar, tokenize
from bayyina.chunking import chunk_text
from bayyina.bm25 import BM25Index
from bayyina.hybrid import rrf
from bayyina.cite import verify_citation, format_citation
from bayyina.types import Chunk, Hit

def test_normalization():
    assert normalize_ar("إِنَّ الـــعقدَ في ٣١") == "ان العقد في 31"
    assert normalize_ar("ؤ ئ") == "ؤ ئ"
    assert tokenize("sha7en iPhone شاحن") == ["sha7en", "iphone", "شاحن"]

@pytest.mark.parametrize("m,o", [(0,0),(10,10),(10,-1)])
def test_invalid_chunk_args(m,o):
    with pytest.raises(ValueError):
        chunk_text("d", "hello world", m, o)

def test_chunk_offsets_and_coverage():
    text = "First paragraph.\n\nSecond paragraph with more words. Third sentence."
    chunks = chunk_text("d", text, 25, 5)
    assert chunks
    for c in chunks:
        assert text[c.start:c.end] == c.text
        assert c.text.strip() == c.text and len(c.text) <= 25
    for i, ch in enumerate(text):
        if not ch.isspace():
            assert any(c.start <= i < c.end for c in chunks)

def test_empty_chunk():
    assert chunk_text("d", "  \n ") == []

def test_bm25_formula_and_tie():
    cs = [Chunk("d","d#b",0,10,"alpha beta"), Chunk("d","d#a",10,20,"alpha beta")]
    ix = BM25Index()
    ix.add(cs)
    result = ix.search("alpha alpha", 5)
    assert [x[0] for x in result] == ["d#a","d#b"]
    assert result[0][1] == pytest.approx(result[1][1])

def test_rrf_deduplicates_per_list():
    scores = dict(rrf([["x","x","y"],["y"]], k=60))
    assert scores["x"] == pytest.approx(1/61)

def test_citation():
    doc = "alpha \n beta gamma"
    assert verify_citation("alpha beta", doc, 0, len(doc))
    assert not verify_citation("alpha beta", doc, -1, len(doc))
    assert not verify_citation(" ", doc, 0, len(doc))

def test_format_citation():
    h = Hit("d#0","d",2,30,"one two three four five",0.5)
    assert format_citation(h, 12) == '[d:2-30] "one two…"'
