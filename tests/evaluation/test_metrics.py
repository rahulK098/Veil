from veil.detect import Detection, Source
from veil.evaluation import Example, GoldSpan, aggregate, score_example
from veil.evaluation.metrics import CleanCorpusMetrics, EntityMetrics


def _det(entity: str, start: int, end: int) -> Detection:
    return Detection(
        entity_type=entity,
        start=start,
        end=end,
        confidence=0.9,
        source=Source.PRESIDIO,
        recognizer="r",
    )


EXAMPLE = Example(
    id="e1",
    text="Call Maria Gonzalez at 555-867-5309 now",
    spans=(
        GoldSpan(entity_type="PERSON", start=5, end=19),
        GoldSpan(entity_type="PHONE_NUMBER", start=23, end=35),
    ),
    category="pii",
)


def test_exact_matches_count_as_tp_and_exact() -> None:
    counts, misses = score_example(EXAMPLE, [_det("PERSON", 5, 19), _det("PHONE_NUMBER", 23, 35)])
    assert counts["PERSON"] == [1, 0, 0, 1]
    assert counts["PHONE_NUMBER"] == [1, 0, 0, 1]
    assert misses == []


def test_partial_overlap_above_iou_is_tp_but_not_exact() -> None:
    counts, _ = score_example(EXAMPLE, [_det("PERSON", 5, 15)])  # "Maria Gonz" -> IoU 10/14
    assert counts["PERSON"] == [1, 0, 0, 0]


def test_partial_overlap_below_iou_is_fn_and_fp() -> None:
    counts, misses = score_example(EXAMPLE, [_det("PERSON", 5, 10)])  # "Maria" -> IoU 5/14
    assert counts["PERSON"] == [0, 1, 1, 0]
    assert {m.kind for m in misses} == {"fn", "fp"}


def test_type_confusion_records_predicted_as() -> None:
    counts, misses = score_example(EXAMPLE, [_det("UK_NHS", 23, 35)])
    assert counts["PHONE_NUMBER"][2] == 1
    assert counts["UK_NHS"][1] == 1
    fn = next(m for m in misses if m.kind == "fn" and m.entity_type == "PHONE_NUMBER")
    assert fn.predicted_as == "UK_NHS" and fn.span_text == "555-867-5309"


def test_each_gold_matches_at_most_one_detection() -> None:
    counts, _ = score_example(EXAMPLE, [_det("PERSON", 5, 19), _det("PERSON", 5, 19)])
    assert counts["PERSON"] == [1, 1, 0, 1]


def test_aggregate_and_derived_metrics() -> None:
    metrics = aggregate([{"A": [3, 1, 1, 2]}, {"A": [1, 0, 1, 1], "B": [0, 0, 2, 0]}])
    by_name = {m.entity_type: m for m in metrics}
    a = by_name["A"]
    assert (a.tp, a.fp, a.fn, a.exact, a.support) == (4, 1, 2, 3, 6)
    assert a.precision == 0.8 and a.recall == 4 / 6 and a.exact_rate == 0.75
    assert round(a.f1, 4) == round(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6), 4)
    b = by_name["B"]
    assert b.precision == 0.0 and b.recall == 0.0 and b.f1 == 0.0 and b.exact_rate == 0.0


def test_entity_metrics_zero_division_safe() -> None:
    m = EntityMetrics(entity_type="X", support=0, tp=0, fp=0, fn=0, exact=0)
    assert (m.precision, m.recall, m.f1, m.exact_rate) == (0.0, 0.0, 0.0, 0.0)


def test_clean_corpus_rates() -> None:
    c = CleanCorpusMetrics(
        documents=10,
        documents_with_any_detection=4,
        detections_by_entity={"PERSON": 5},
        documents_modified={"logs": 3},
        documents_blocked={},
    )
    assert c.detection_fp_rate == 0.4
    assert c.modified_rate("logs") == 0.3 and c.modified_rate("model") == 0.0
    empty = CleanCorpusMetrics(
        documents=0,
        documents_with_any_detection=0,
        detections_by_entity={},
        documents_modified={},
        documents_blocked={},
    )
    assert empty.detection_fp_rate == 0.0 and empty.modified_rate("logs") == 0.0
