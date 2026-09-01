"""Tests for domain balancing and the participant-level split.

Both modules exist to protect claims rather than code. Balancing protects "a small
model specialised on three domains"; the split protects "the validation NLL measures
generalisation". A silent bug in either makes the corresponding sentence in the thesis
false while every script still exits zero.
"""

from __future__ import annotations

from centauro_lite.core.sampling import ParticipantRef, balance_domains, choices_per_domain
from centauro_lite.core.splits import (
    leaked_participants,
    load_manifest,
    manifest,
    split_participants,
    split_summary,
)


def _refs(experiment: str, domain: str, count: int, choices: int) -> list[ParticipantRef]:
    """Build a run of identical participants for one experiment."""
    return [
        ParticipantRef(
            experiment=experiment, participant=f"p{index}", domain=domain, n_choices=choices
        )
        for index in range(count)
    ]


def test_the_dominant_experiment_stops_dominating() -> None:
    """Without a cap, peterson2021using is ~92% of the mix and specialisation is untestable."""
    refs = _refs("peterson/e.csv", "risky_choice", 1000, 80)
    refs += _refs("badham/e.csv", "categorization", 85, 350)
    totals = choices_per_domain(balance_domains(refs, max_choices_per_domain=20_000, seed=1))
    assert abs(totals["risky_choice"] - totals["categorization"]) < 1000


def test_a_domain_smaller_than_the_cap_is_kept_whole() -> None:
    """Balancing may shrink the large domains; it must never invent or drop small ones."""
    refs = _refs("small/e.csv", "categorization", 10, 100)
    totals = choices_per_domain(balance_domains(refs, max_choices_per_domain=50_000, seed=1))
    assert totals == {"categorization": 1000}


def test_experiments_inside_a_domain_are_balanced_too() -> None:
    """kool2016when has two files; one must not swallow the domain's whole budget."""
    refs = _refs("kool/exp1.csv", "reinforcement_learning", 500, 120)
    refs += _refs("kool/exp2.csv", "reinforcement_learning", 500, 240)
    selected = balance_domains(refs, max_choices_per_domain=12_000, seed=1)
    per_experiment = {ref.experiment for ref in selected}
    counts = {name: sum(1 for ref in selected if ref.experiment == name) for name in per_experiment}
    assert len(counts) == 2
    assert min(counts.values()) > 0


def test_no_cap_keeps_everything() -> None:
    """Evaluation runs on the full data; the cap is a training-mix decision only."""
    refs = _refs("a/e.csv", "risky_choice", 20, 100)
    assert len(balance_domains(refs, max_choices_per_domain=None, seed=1)) == 20


def test_balancing_is_reproducible() -> None:
    """A different selection between runs makes two results incomparable."""
    refs = _refs("a/e.csv", "risky_choice", 200, 50)
    first = balance_domains(refs, max_choices_per_domain=1000, seed=7)
    second = balance_domains(refs, max_choices_per_domain=1000, seed=7)
    assert first == second


def test_a_different_seed_selects_differently() -> None:
    """Otherwise the seed is decorative and the sampling is not really randomised."""
    refs = _refs("a/e.csv", "risky_choice", 200, 50)
    assert balance_domains(refs, max_choices_per_domain=1000, seed=1) != balance_domains(
        refs, max_choices_per_domain=1000, seed=2
    )


def test_no_participant_is_in_both_splits() -> None:
    """The leak this guards against turns validation into a memorisation test."""
    keys = [("exp/a.csv", f"p{index}") for index in range(100)]
    assignment = split_participants(keys, val_fraction=0.1, seed=3407)
    assert leaked_participants(assignment) == set()
    assert len(assignment.keys) == 100


def test_every_experiment_reaches_validation() -> None:
    """An experiment absent from validation cannot be reported on, and absence is quiet."""
    keys = [("big/a.csv", f"p{index}") for index in range(100)]
    keys += [("tiny/a.csv", "only")]
    keys += [("small/a.csv", "p0"), ("small/a.csv", "p1")]
    assignment = split_participants(keys, val_fraction=0.1, seed=3407)
    validated = {experiment for experiment, _ in assignment.validation}
    assert validated == {"big/a.csv", "tiny/a.csv", "small/a.csv"}


def test_training_is_not_emptied_by_rounding() -> None:
    """Holding out the only participant of a two-person experiment would leave no data."""
    keys = [("small/a.csv", "p0"), ("small/a.csv", "p1")]
    assignment = split_participants(keys, val_fraction=0.9, seed=1)
    assert len(assignment.train) == 1
    assert len(assignment.validation) == 1


def test_split_is_reproducible() -> None:
    """Evaluation must be able to prove it ran on the same validation set as training."""
    keys = [("exp/a.csv", f"p{index}") for index in range(50)]
    assert split_participants(keys, val_fraction=0.2, seed=42) == split_participants(
        keys, val_fraction=0.2, seed=42
    )


def test_summary_counts_both_sides_per_experiment() -> None:
    """The per-experiment breakdown is how an unbalanced split gets noticed."""
    keys = [("a/x.csv", f"p{index}") for index in range(10)]
    keys += [("b/x.csv", f"q{index}") for index in range(10)]
    summary = split_summary(split_participants(keys, val_fraction=0.2, seed=1))
    assert summary["a/x.csv"] == {"train": 8, "validation": 2}
    assert summary["b/x.csv"] == {"train": 8, "validation": 2}


def test_manifest_round_trips() -> None:
    """Reloading beats recomputing: a library upgrade can change the underlying order.

    "Seed 3407" only reproduces a permutation of whatever order the dataset had. The
    manifest is the record that actually survives.
    """
    keys = [("exp/a.csv", f"p{index}") for index in range(30)]
    assignment = split_participants(keys, val_fraction=0.1, seed=3407)
    assert load_manifest(manifest(assignment)) == assignment
