import pytest

from embedding_utils import chunk_text, stable_embedding_id, vector_literal


def test_short_text_is_one_normalized_chunk():
    assert chunk_text("  Flooding\n is   possible. ") == ["Flooding is possible."]


def test_sliding_window_preserves_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    assert chunks[0][-3:] == chunks[1][:3]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("weather", size, overlap)


def test_embedding_id_is_deterministic_and_chunk_specific():
    first = stable_embedding_id("doc-1", "mini-lm", 0)
    assert first == stable_embedding_id("doc-1", "mini-lm", 0)
    assert first != stable_embedding_id("doc-1", "mini-lm", 1)


def test_vector_literal_uses_pgvector_format():
    assert vector_literal([1, 0.25, -3.5]) == "[1,0.25,-3.5]"

