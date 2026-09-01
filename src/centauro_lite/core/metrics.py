"""Aggregating the negative log-likelihood over human choices.

This is the number the whole project is judged on, so the arithmetic lives here in
plain Python where it can be tested, separate from the GPU code that feeds it.

Two decisions shape the result:

*Weighting by token.* Participants contribute wildly different numbers of choices, so
averaging the per-batch losses would give a participant with 40 choices the same weight
as one with 400. The aggregate is the total loss divided by the total scored tokens.

*Counting after the shift.* A causal model predicts token ``i`` from everything before
it, so Hugging Face drops position 0 when computing the loss. A window that happens to
begin on a scored token would therefore report one more scored token than actually
entered the loss, and the denominator would be quietly wrong.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from centauro_lite.core.chunking import IGNORE_INDEX

CENTAUR_NLL = 0.44
"""Centaur (Llama 3.1 70B + QLoRA) on the full Psych-101 test set, from the paper."""

LLAMA_BASE_NLL = 0.58
"""Llama 3.1 70B without fine-tuning, from the same table."""

COGNITIVE_MODELS_NLL = 0.56
"""Mean of 14 specialised cognitive models, from the same table."""


def scored_token_count(labels: Sequence[Sequence[int]]) -> int:
    """Count the label positions that actually enter the loss.

    Args:
        labels: Label ids per example, padded with :data:`IGNORE_INDEX`.

    Returns:
        Number of scored positions, excluding index 0 of each example.

    Note:
        Excluding index 0 is not a rounding detail. It is the difference between the
        denominator the metric reports and the denominator the model actually used,
        and nothing in the stack would complain if the two disagreed.
    """
    return sum(1 for row in labels for label in row[1:] if label != IGNORE_INDEX)


class NllResult(BaseModel):
    """A finished evaluation, ready to serialise next to the paper's references."""

    model_config = ConfigDict(frozen=True)

    label: str
    nll: float
    n_scored_tokens: int
    per_experiment: dict[str, float]
    per_experiment_tokens: dict[str, int]

    def comparison(self) -> dict[str, float]:
        """Return this result alongside the published reference values.

        Returns:
            The measured NLL plus the paper's numbers.

        Note:
            The references were measured on the full 160-experiment test set with a
            different tokenizer. They are a landmark, not a like-for-like comparison;
            the like-for-like one is Minitaur-8B run through this same code on this
            same validation split.
        """
        return {
            self.label: self.nll,
            "reference_centaur_70b": CENTAUR_NLL,
            "reference_cognitive_models": COGNITIVE_MODELS_NLL,
            "reference_llama_base": LLAMA_BASE_NLL,
        }


class NllAccumulator:
    """Collects loss across batches and experiments without losing the weighting."""

    def __init__(self) -> None:
        self._loss_sum: dict[str, float] = defaultdict(float)
        self._tokens: dict[str, int] = defaultdict(int)

    def add(self, experiment: str, mean_loss: float, n_scored: int) -> None:
        """Record one batch.

        Args:
            experiment: Experiment the batch belongs to, so per-experiment NLL can be
                reported. The aggregate alone hides a model that is excellent at one
                task and useless at another, which is exactly the finding worth having.
            mean_loss: The model's mean loss over the scored tokens of this batch.
            n_scored: Scored tokens in the batch, counted after the causal shift.

        Note:
            Batches with no scored tokens are ignored rather than treated as zero loss.
            Averaging in a zero would drag the result down for free.
        """
        if n_scored <= 0:
            return
        self._loss_sum[experiment] += mean_loss * n_scored
        self._tokens[experiment] += n_scored

    @property
    def n_scored_tokens(self) -> int:
        """Total scored tokens seen so far.

        Returns:
            Sum across experiments.
        """
        return sum(self._tokens.values())

    def per_experiment(self) -> dict[str, float]:
        """NLL for each experiment separately.

        Returns:
            Experiment name to NLL.
        """
        return {
            experiment: self._loss_sum[experiment] / tokens
            for experiment, tokens in self._tokens.items()
            if tokens
        }

    def per_experiment_tokens(self) -> dict[str, int]:
        """Scored tokens behind each per-experiment NLL.

        Returns:
            Experiment name to token count. Without this, a per-experiment NLL built
            on a handful of tokens looks as authoritative as one built on thousands.
        """
        return dict(self._tokens)

    def result(self, label: str) -> NllResult:
        """Finalise the aggregate.

        Args:
            label: Name of what was evaluated, such as ``"qwen3-1.7b-base"``.

        Returns:
            The completed result.

        Raises:
            ValueError: When nothing was scored. An empty evaluation almost always
                means the masking or the split is broken, and returning ``0.0`` -- a
                perfect score -- would be the worst possible way to report that.
        """
        total_tokens = self.n_scored_tokens
        if total_tokens == 0:
            msg = "No scored tokens: the masking or the validation split is empty"
            raise ValueError(msg)

        return NllResult(
            label=label,
            nll=sum(self._loss_sum.values()) / total_tokens,
            n_scored_tokens=total_tokens,
            per_experiment=self.per_experiment(),
            per_experiment_tokens=self.per_experiment_tokens(),
        )
