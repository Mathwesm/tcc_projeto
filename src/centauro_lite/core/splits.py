"""Dividing participants between training and validation.

The unit of the split is the **participant**, never the window and never the row. Two
windows of the same session landing on opposite sides leak that participant's history
from training into validation, and validation stops measuring generalisation and starts
measuring memorisation. Because windows are produced *after* this split, that leak is
structurally impossible rather than merely avoided.

The split is stratified by experiment so every domain appears on both sides. Without
that, a small experiment can end up entirely in training and its validation NLL becomes
undefined -- silently, as an absent row rather than an error.

The chosen participant ids are written to disk. "Seed 3407" is not reproducible on its
own: it reproduces a *permutation of whatever order the dataset happened to have*, and
that order can change between library versions. The manifest is the actual record.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import NamedTuple


class SplitAssignment(NamedTuple):
    """Which participants belong to each side of the split."""

    train: tuple[tuple[str, str], ...]
    validation: tuple[tuple[str, str], ...]

    @property
    def keys(self) -> set[tuple[str, str]]:
        """Every participant key in the assignment.

        Returns:
            Union of both sides.
        """
        return set(self.train) | set(self.validation)


def split_participants(
    keys: Iterable[tuple[str, str]],
    *,
    val_fraction: float,
    seed: int,
) -> SplitAssignment:
    """Assign ``(experiment, participant)`` keys to training or validation.

    Args:
        keys: Participant keys to divide.
        val_fraction: Share held out for validation, applied per experiment.
        seed: Seed for the deterministic shuffle.

    Returns:
        The assignment, with both sides sorted for a stable order downstream.

    Note:
        Every experiment contributes at least one validation participant, even when
        ``val_fraction`` rounds down to zero for it. A domain missing from validation
        cannot be reported on, and that absence is far easier to miss than a small
        validation set.
    """
    rng = random.Random(seed)  # noqa: S311 - splitting data, not cryptography
    by_experiment: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(keys):
        by_experiment[key[0]].append(key)

    train: list[tuple[str, str]] = []
    validation: list[tuple[str, str]] = []
    for experiment_keys in by_experiment.values():
        shuffled = list(experiment_keys)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        validation.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    return SplitAssignment(train=tuple(sorted(train)), validation=tuple(sorted(validation)))


def leaked_participants(assignment: SplitAssignment) -> set[tuple[str, str]]:
    """Return participants present on both sides of the split.

    This is the cheapest guard against the most expensive mistake in the project, and
    it costs one set intersection. Call it before writing anything to disk.

    Args:
        assignment: The split to check.

    Returns:
        The offending keys, empty when the split is clean.
    """
    return set(assignment.train) & set(assignment.validation)


def split_summary(assignment: SplitAssignment) -> dict[str, dict[str, int]]:
    """Count participants per experiment on each side.

    Args:
        assignment: The split to summarise.

    Returns:
        Per-experiment counts under ``train`` and ``validation`` keys.
    """
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0})
    for experiment, _ in assignment.train:
        summary[experiment]["train"] += 1
    for experiment, _ in assignment.validation:
        summary[experiment]["validation"] += 1
    return dict(summary)


def manifest(
    assignment: SplitAssignment, extra: dict[str, object] | None = None
) -> dict[str, object]:
    """Build the on-disk record of who ended up where.

    Args:
        assignment: The split to record.
        extra: Additional provenance to store alongside, such as the configuration
            that produced the split.

    Returns:
        A JSON-serialisable mapping.
    """
    record: dict[str, object] = {
        "train": [list(key) for key in assignment.train],
        "validation": [list(key) for key in assignment.validation],
        "counts": {
            "train": len(assignment.train),
            "validation": len(assignment.validation),
        },
        "per_experiment": split_summary(assignment),
    }
    if extra:
        record.update(extra)
    return record


def load_manifest(record: dict[str, object]) -> SplitAssignment:
    """Rebuild an assignment from a manifest.

    Reloading rather than recomputing is what makes a later evaluation provably run on
    the same validation set as the training it is judging.

    Args:
        record: A mapping produced by :func:`manifest`.

    Returns:
        The stored assignment.
    """

    def _keys(name: str) -> tuple[tuple[str, str], ...]:
        raw = record[name]
        if not isinstance(raw, Sequence):
            msg = f"Malformed manifest: '{name}' is not a list"
            raise TypeError(msg)
        return tuple((str(item[0]), str(item[1])) for item in raw)

    return SplitAssignment(train=_keys("train"), validation=_keys("validation"))
