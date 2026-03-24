from __future__ import annotations

from ..config import ScoringConfig
from ..interfaces import EventScorer
from ..schemas import PackedFeature, ScoreResult, TemporalOutput


class CompositeEventScorer(EventScorer):
    """context/boundary/uncertainty/novelty를 결합하는 baseline scorer."""

    def __init__(self, cfg: ScoringConfig) -> None:
        self._cfg = cfg
        self._prev_vector: list[float] | None = None

    def score(self, packed: PackedFeature, temporal: TemporalOutput) -> ScoreResult:
        context_score = max(temporal.context_probs.values()) if temporal.context_probs else 0.0
        boundary_score = temporal.boundary_signal
        uncertainty_score = 1.0 - context_score
        novelty_score = self._compute_novelty(packed.spatial_vector)

        total_importance = (
            (self._cfg.context_weight * context_score)
            + (self._cfg.boundary_weight * boundary_score)
            + (self._cfg.uncertainty_weight * uncertainty_score)
            + (self._cfg.novelty_weight * novelty_score)
        )
        total_importance = max(0.0, min(1.0, total_importance))

        self._prev_vector = packed.spatial_vector[:]
        return ScoreResult(
            frame_id=packed.frame_id,
            context_score=round(context_score, 6),
            boundary_score=round(boundary_score, 6),
            uncertainty_score=round(uncertainty_score, 6),
            novelty_score=round(novelty_score, 6),
            total_importance=round(total_importance, 6),
        )

    def _compute_novelty(self, vector: list[float]) -> float:
        if not vector or self._prev_vector is None:
            return 0.0
        length = min(len(vector), len(self._prev_vector))
        if length == 0:
            return 0.0
        delta = sum(abs(vector[idx] - self._prev_vector[idx]) for idx in range(length)) / length
        return max(0.0, min(1.0, delta))
