from app.retrieval import dedupe


def test_dedupe_removes_same_text_keeps_best_score():
    results = [
        {"text": "[DESCRIPTION|hin] बाइनरी सर्च एक विधि है", "score": 0.5, "strategy": "metadata_raw"},
        {"text": "[DESCRIPTION|hin] बाइनरी सर्च एक विधि है", "score": 0.7, "strategy": "fixed512"},
        {"text": "मानसून की बारिश खेती के लिए महत्वपूर्ण है", "score": 0.6, "strategy": "semantic"},
    ]
    out = dedupe(results)
    assert len(out) == 2
    assert out[0]["score"] == 0.7


def test_dedupe_distinct_texts_untouched():
    results = [
        {"text": "alpha beta gamma", "score": 0.9},
        {"text": "delta epsilon zeta", "score": 0.4},
    ]
    assert len(dedupe(results)) == 2


def test_dedupe_empty():
    assert dedupe([]) == []
