from app.ingestion.normalizer import normalize_text


def test_normalizer_cleans_whitespace_and_nulls() -> None:
    text = "Alpha\x00   beta\r\n\r\n\r\nGamma"

    assert normalize_text(text) == "Alpha beta\n\nGamma"


def test_normalizer_preserves_paragraph_boundaries() -> None:
    text = "Heading\n\nFirst paragraph.\nSecond line."

    assert normalize_text(text) == "Heading\n\nFirst paragraph.\nSecond line."
