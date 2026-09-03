"""Trivial predictors, to say how much of the score is actually about behaviour.

An NLL of 0.5388 means nothing on its own. Part of it is available to anything that
counts: if participants press D twice as often as P, a model that has learned only that
already scores well without understanding a single trial.

Three predictors bracket the trivial, and the gaps between them are the interesting
part.

**Uniform** spreads probability evenly over every answer an experiment uses.
**Marginal** weights those answers by how often they appear in training. Both are
useless here, and usefully so: Psych-101 assigns the response keys at random per
participant, so an experiment uses all 26 letters while any one session uses about
three. Counting letters therefore learns nothing -- and the measurement says so, since
marginal comes out *worse* than uniform, its estimated frequencies being noise.

**Informed uniform** is the floor that matters. It knows which keys a session actually
offers and guesses evenly among them, which is what a reader who understood the
instructions and nothing else would do. Anything below it is knowledge about the person,
not about the task.

All three run on the prepared dataset in seconds on a CPU, and they turn "0.5388" into
"closed X% of the distance between understanding the task and the reference" -- a claim
about the work rather than about the alphabet.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from centauro_lite.core.chunking import IGNORE_INDEX
from centauro_lite.core.metrics import NllAccumulator, NllResult

SMOOTHING = 1.0
"""Add-one smoothing. An answer seen in validation but never in training would
otherwise cost infinite loss and destroy the aggregate -- a single rare keypress would
decide the number."""


class LabelledWindow(NamedTuple):
    """One prepared window: whose session it is, and its label row."""

    experiment: str
    participant: str
    labels: list[int]


def scored_labels(window: LabelledWindow) -> Iterator[int]:
    """Yield the label tokens that actually enter the loss.

    Args:
        window: A prepared window.

    Yields:
        Each scored token id.

    Note:
        Index 0 is skipped, exactly as in :mod:`centauro_lite.core.metrics`. A causal
        model cannot predict the first position of a window, so it is excluded there;
        including it here would score the trivial predictors on a slightly larger set
        than the neural one and quietly tilt the comparison.
    """
    for label in window.labels[1:]:
        if label != IGNORE_INDEX:
            yield label


def answer_counts(windows: Iterable[LabelledWindow]) -> dict[str, Counter[int]]:
    """Count how often each answer token occurs, per experiment.

    Args:
        windows: Training windows.

    Returns:
        Answer frequencies keyed by experiment. Counting per experiment rather than
        globally is what keeps this a *fair* trivial model: the keys differ from task to
        task, and pooling them would handicap it for no good reason.
    """
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for window in windows:
        counts[window.experiment].update(scored_labels(window))
    return dict(counts)


def alternatives_per_participant(
    windows: Iterable[LabelledWindow],
) -> dict[tuple[str, str], set[int]]:
    """Collect the answer keys each session actually uses.

    Args:
        windows: Windows to scan.

    Returns:
        The distinct answer tokens per ``(experiment, participant)``.

    Note:
        This reads the keys from the answers themselves, so an option a participant
        never chose does not appear and the set can be too small. The resulting floor is
        therefore slightly *too easy to beat*, which errs in the conservative direction:
        it understates how much the model added rather than overstating it.
    """
    keys: dict[tuple[str, str], set[int]] = defaultdict(set)
    for window in windows:
        keys[window.experiment, window.participant].update(scored_labels(window))
    return dict(keys)


def informed_uniform_result(windows: Iterable[LabelledWindow]) -> NllResult:
    """Score a predictor that knows the response options and nothing else.

    Args:
        windows: Validation windows.

    Returns:
        The NLL of guessing evenly among the keys a session offers -- what someone who
        read the instructions and learned nothing about the participant would score.

    Note:
        Unlike the other two, this needs no training data: the response options are a
        property of the task, not of behaviour. That is also what makes it the honest
        floor, since nothing about the held-out people leaks into it beyond which
        buttons existed.
    """
    windows = list(windows)
    options = alternatives_per_participant(windows)
    accumulator = NllAccumulator()

    for window in windows:
        n_alternatives = max(1, len(options[window.experiment, window.participant]))
        surprise = math.log(n_alternatives)
        for _ in scored_labels(window):
            accumulator.add(window.experiment, surprise, 1)

    return accumulator.result("trivial-informed")


def _score(
    windows: Iterable[LabelledWindow],
    counts: dict[str, Counter[int]],
    *,
    label: str,
    use_frequencies: bool,
) -> NllResult:
    """Score validation windows under a context-free predictor.

    Args:
        windows: Validation windows.
        counts: Answer frequencies from :func:`answer_counts`.
        label: Name for the result.
        use_frequencies: ``True`` for the marginal predictor, ``False`` for uniform.

    Returns:
        The NLL, aggregated exactly as the neural evaluation aggregates it.
    """
    accumulator = NllAccumulator()

    for window in windows:
        experiment = window.experiment
        observed = counts.get(experiment, Counter())
        # An unseen answer still needs somewhere to live, hence the extra slot.
        n_alternatives = len(observed) + 1
        total = sum(observed.values()) + SMOOTHING * n_alternatives

        for token in scored_labels(window):
            if use_frequencies:
                probability = (observed[token] + SMOOTHING) / total
            else:
                probability = 1.0 / n_alternatives
            accumulator.add(experiment, -math.log(probability), 1)

    return accumulator.result(label)


def uniform_result(
    train: Iterable[LabelledWindow], validation: Iterable[LabelledWindow]
) -> NllResult:
    """Score the uniform predictor: every answer of an experiment equally likely.

    Args:
        train: Training windows, used only to learn which answers an experiment uses.
        validation: Windows to score.

    Returns:
        The NLL of pure ignorance about which answers people prefer.
    """
    return _score(validation, answer_counts(train), label="trivial-uniform", use_frequencies=False)


def marginal_result(
    train: Iterable[LabelledWindow], validation: Iterable[LabelledWindow]
) -> NllResult:
    """Score the marginal predictor: answer frequencies, ignoring all context.

    Args:
        train: Training windows, used to estimate the frequencies.
        validation: Windows to score.

    Returns:
        The NLL of the best context-free predictor.
    """
    return _score(validation, answer_counts(train), label="trivial-marginal", use_frequencies=True)


def recovered_fraction(trivial: float, model: float, reference: float) -> float:
    """Share of the reachable gap that a model actually closed.

    Args:
        trivial: NLL of the trivial predictor -- where counting alone gets you.
        model: NLL of the model being judged.
        reference: NLL of the best comparison point, taken as the practical target.

    Returns:
        Percentage of the distance from ``trivial`` to ``reference`` that ``model``
        covered. Above 100 means the model passed the reference.

    Note:
        This reframes an absolute score as progress along a range whose ends are both
        measured. "0.5388" invites comparison with a number from another test set;
        "closed 82% of the distance between counting and the reference" does not.
    """
    span = trivial - reference
    if span <= 0:
        msg = f"The reference ({reference}) must be better than the trivial floor ({trivial})"
        raise ValueError(msg)
    return 100.0 * (trivial - model) / span
