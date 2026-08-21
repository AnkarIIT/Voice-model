from app.preprocess import clean_text, normalize_passages


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Hello</p>   world ") == "Hello world"


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_normalize_passages_fields():
    row = {
        "query_id": "q1",
        "query_type": "CAUSE",
        "query": " why? ",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "English_passages": ["a", "b"],
            "Translated_passages": ["x", None],
            "is_selected": [1],
        },
    }
    out = normalize_passages(row, use_translated=True)
    assert len(out) == 2
    assert out[0]["doc_id"] == "q1_0"
    assert out[0]["is_selected"] == 1
    assert out[1]["is_selected"] == 0
    assert out[1]["text"] == ""


def test_normalize_passages_missing_keys():
    row = {"query_id": "q2", "query_type": "T", "passages": {}}
    assert normalize_passages(row) == []
