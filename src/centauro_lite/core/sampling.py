"""Balancing the training mix across cognitive domains.

The raw sizes are wildly uneven: ``peterson2021using`` alone carries 1,097,375 choices
against 29,776 in ``badham2017deficits``. Training on that mix produces a model of
``peterson2021using`` with two appendices, and the claim this project actually wants to
test -- that a small model specialised on a few domains competes with a large generalist
*on those domains* -- would be untestable.

Balancing happens on **choices**, not participants, because the loss is computed per
choice. Two experiments with the same participant count can differ threefold in how
many choices each participant contributes.

Selection is at participant granularity and deterministic given the seed: a participant
is taken whole or not at all, so no session is ever split across the sampling boundary.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import NamedTuple


class ParticipantRef(NamedTuple):
    """Enough of a participant to decide whether to keep them, without their text."""

    experiment: str
    participant: str
    domain: str
    n_choices: int


def _round_robin_within_domain(
    by_experiment: dict[str, list[ParticipantRef]],
    cap: int,
) -> list[ParticipantRef]:
    """Take participants one at a time across experiments until the cap is reached.

    Round-robin rather than a proportional quota because it balances the experiments
    inside a domain without needing to know their sizes in advance, and it degrades
    gracefully when one experiment runs out before the others.

    Args:
        by_experiment: Shuffled participants grouped by experiment.
        cap: Maximum number of choices to accumulate for this domain.

    Returns:
        The selected participants.
    """
    cursors = dict.fromkeys(by_experiment, 0)
    selected: list[ParticipantRef] = []
    total = 0

    while total < cap:
        progressed = False
        for experiment, refs in by_experiment.items():
            index = cursors[experiment]
            if index >= len(refs):
                continue
            ref = refs[index]
            cursors[experiment] = index + 1
            selected.append(ref)
            total += ref.n_choices
            progressed = True
            if total >= cap:
                break
        if not progressed:  # every experiment exhausted before reaching the cap
            break

    return selected


def balance_domains(
    refs: Iterable[ParticipantRef],
    *,
    max_choices_per_domain: int | None,
    seed: int,
) -> list[ParticipantRef]:
    """Subsample participants so every domain contributes comparable evidence.

    Args:
        refs: Every candidate participant, already tagged with a domain.
        max_choices_per_domain: Choice budget per domain. ``None`` keeps everything,
            which is the right setting when reporting per-experiment metrics on the
            full data rather than training on it.
        seed: Seed for the deterministic shuffle. Two runs with the same seed select
            the same participants, which is what makes a result reproducible.

    Returns:
        The selected participants, sorted for a stable order downstream.
    """
    refs = list(refs)
    if max_choices_per_domain is None:
        return sorted(refs)

    rng = random.Random(seed)  # noqa: S311 - sampling, not cryptography
    by_domain: dict[str, dict[str, list[ParticipantRef]]] = defaultdict(lambda: defaultdict(list))
    for ref in refs:
        by_domain[ref.domain][ref.experiment].append(ref)

    selected: list[ParticipantRef] = []
    for by_experiment in by_domain.values():
        for experiment_refs in by_experiment.values():
            rng.shuffle(experiment_refs)
        selected.extend(_round_robin_within_domain(by_experiment, max_choices_per_domain))

    return sorted(selected)


def choices_per_domain(refs: Sequence[ParticipantRef]) -> dict[str, int]:
    """Total choices per domain, for reporting how balanced the mix ended up.

    Args:
        refs: Selected participants.

    Returns:
        Choice counts keyed by domain.
    """
    totals: dict[str, int] = defaultdict(int)
    for ref in refs:
        totals[ref.domain] += ref.n_choices
    return dict(totals)
