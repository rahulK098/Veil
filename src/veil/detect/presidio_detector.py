"""Presidio adapter: PII detection via pattern recognizers + spaCy NER (ADR-0001)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from veil.detect.custom import PatternSpec, extended_recognizers
from veil.detect.schema import Detection, Source

logger = logging.getLogger(__name__)

# Entities Presidio can emit that are noisy enough with spaCy `lg` that we do not request them
# unless a policy explicitly asks. ORGANIZATION fires on tokens like "SSN" and digit runs;
# DATE_TIME fires on nearly any number group and is rarely governed PII.
DEFAULT_EXCLUDED_ENTITIES: frozenset[str] = frozenset({"ORGANIZATION", "DATE_TIME", "URL", "NRP"})

DEFAULT_MODEL = "en_core_web_lg"


def build_analyzer(
    model_name: str = DEFAULT_MODEL,
    custom_specs: Sequence[PatternSpec] = (),
    *,
    extended: bool = True,
) -> AnalyzerEngine:
    """Construct a Presidio analyzer. Expensive (~4 s); call once per process.

    ``extended`` adds Veil's recognizers for documented Presidio gaps (see ``custom.py``).
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    for spec in custom_specs:
        engine.registry.add_recognizer(spec.to_recognizer())
    if extended:
        for recognizer in extended_recognizers():
            engine.registry.add_recognizer(recognizer)
    return engine


class PresidioDetector:
    """Wraps a shared ``AnalyzerEngine`` and normalizes its results."""

    def __init__(
        self,
        analyzer: AnalyzerEngine,
        *,
        entities: Sequence[str] | None = None,
        excluded_entities: frozenset[str] = DEFAULT_EXCLUDED_ENTITIES,
        custom_recognizer_names: frozenset[str] = frozenset(),
    ) -> None:
        self._analyzer = analyzer
        self._custom_recognizer_names = custom_recognizer_names
        supported = set(analyzer.get_supported_entities(language="en"))
        requested = set(entities) if entities is not None else supported - excluded_entities
        unknown = requested - supported
        if unknown:
            raise ValueError(f"unsupported Presidio entities requested: {sorted(unknown)}")
        self._entities = sorted(requested)

    @property
    def name(self) -> str:
        return "presidio"

    @property
    def entities(self) -> list[str]:
        return list(self._entities)

    def detect(self, text: str) -> Sequence[Detection]:
        if not text:
            return []
        results: list[RecognizerResult] = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            score_threshold=0.0,
            return_decision_process=True,
        )
        return [self._to_detection(r) for r in results]

    def _to_detection(self, result: RecognizerResult) -> Detection:
        explanation = result.analysis_explanation
        recognizer = explanation.recognizer if explanation is not None else "unknown"
        source = Source.CUSTOM if recognizer in self._custom_recognizer_names else Source.PRESIDIO
        return Detection(
            entity_type=result.entity_type,
            start=result.start,
            end=result.end,
            confidence=min(max(result.score, 0.0), 1.0),
            source=source,
            recognizer=recognizer,
        )
