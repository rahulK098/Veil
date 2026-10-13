from veil.detect.merge import resolve_overlaps
from veil.detect.schema import Detection, Source


def _det(
    entity: str, start: int, end: int, conf: float, source: Source = Source.PRESIDIO
) -> Detection:
    return Detection(
        entity_type=entity,
        start=start,
        end=end,
        confidence=conf,
        source=source,
        recognizer="r",
    )


def test_returns_sorted_by_start_when_no_overlap() -> None:
    dets = [_det("B", 10, 12, 0.5), _det("A", 0, 3, 0.5)]
    assert [d.entity_type for d in resolve_overlaps(dets)] == ["A", "B"]


def test_higher_confidence_wins_overlap() -> None:
    email = _det("EMAIL_ADDRESS", 0, 19, 1.0)
    url_fragment = _det("URL", 8, 19, 0.5)
    assert resolve_overlaps([url_fragment, email]) == [email]


def test_longer_span_breaks_confidence_tie() -> None:
    short = _det("PERSON", 0, 4, 0.85)
    long = _det("LOCATION", 0, 10, 0.85)
    assert resolve_overlaps([short, long]) == [long]


def test_source_priority_breaks_full_tie() -> None:
    presidio = _det("IP_ADDRESS", 0, 8, 0.9, Source.PRESIDIO)
    secrets = _det("IP_ADDRESS", 0, 8, 0.9, Source.DETECT_SECRETS)
    custom = _det("IP_ADDRESS", 0, 8, 0.9, Source.CUSTOM)
    assert resolve_overlaps([presidio, secrets, custom]) == [custom]


def test_chain_of_overlaps_is_resolved_greedily() -> None:
    a = _det("A", 0, 5, 0.9)
    b = _det("B", 4, 9, 0.95)  # overlaps a and c
    c = _det("C", 8, 12, 0.9)
    # b wins first, knocking out both a and c
    assert resolve_overlaps([a, b, c]) == [b]


def test_empty_input() -> None:
    assert resolve_overlaps([]) == []
