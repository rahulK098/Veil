from pathlib import Path

from veil.evaluation import Example, generate_clean, generate_labelled, read_jsonl, write_jsonl


def test_generate_labelled_meets_ship_gate_size() -> None:
    examples = generate_labelled()
    assert len(examples) >= 150
    types = {s.entity_type for ex in examples for s in ex.spans}
    assert {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "STREET_ADDRESS",
        "CREDIT_CARD",
        "AWS_ACCESS_KEY",
        "GITHUB_TOKEN",
        "PRIVATE_KEY",
    } <= types


def test_gold_spans_have_exact_offsets() -> None:
    """Every span must slice to a value that looks like its entity — offsets are computed,
    never guessed, and this guards the builder against drift."""
    for ex in generate_labelled():
        for span in ex.spans:
            value = ex.span_text(span)
            assert value == value.strip(), (ex.id, span)
            if span.entity_type == "EMAIL_ADDRESS":
                assert "@" in value
            if span.entity_type == "AWS_ACCESS_KEY":
                assert value.startswith("AKIA") and len(value) == 20
            if span.entity_type == "US_SSN":
                assert len(value) == 11 and value[3] == "-"
            if span.entity_type == "PRIVATE_KEY":
                assert value.startswith("-----BEGIN") and value.endswith("KEY-----")


def test_generation_is_deterministic_per_seed() -> None:
    assert generate_labelled(seed=7) == generate_labelled(seed=7)
    assert generate_labelled(seed=7) != generate_labelled(seed=8)


def test_clean_corpus_has_no_spans_and_uses_real_business_shapes() -> None:
    clean = generate_clean(count=100)
    assert len(clean) == 100
    assert all(ex.spans == () and ex.category == "clean" for ex in clean)
    assert {ex.kind for ex in clean} >= {"contract", "email", "invoice", "policy"}


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = generate_labelled(pii=3, secrets=2, mixed=1)
    path = tmp_path / "x.jsonl"
    write_jsonl(examples, path)
    loaded = read_jsonl(path)
    assert loaded == examples
    assert all(isinstance(ex, Example) for ex in loaded)
