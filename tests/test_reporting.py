"""Tests for the comparison table.

The table is what the thesis argues from, so the failure that matters here is not a
crash: it is a row that looks comparable to its neighbours and is not.
"""

from __future__ import annotations

import pytest

from centauro_lite.core.reporting import (
    ReportRow,
    format_table,
    is_baseline,
    load_rows,
    rank,
    with_improvements,
)


def _row(label: str, nll: float, fingerprint: str = "abc123", tokens: int = 8700) -> ReportRow:
    """Build a row for the table."""
    return ReportRow(label=label, nll=nll, n_scored_tokens=tokens, fingerprint=fingerprint)


def test_metadata_keys_are_not_measurements() -> None:
    """The results file mixes measurements with metadata under underscore keys."""
    rows = load_rows(
        {
            "qwen3-base": {"nll": 0.92, "n_scored_tokens": 8700},
            "_references": {"reference_centaur_70b": 0.44},
        }
    )
    assert [row.label for row in rows] == ["qwen3-base"]


def test_lower_nll_ranks_first() -> None:
    """Lower means the model was less surprised by what the human did."""
    ordered = rank([_row("worse", 0.92), _row("better", 0.64)])
    assert [row.label for row in ordered] == ["better", "worse"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("qwen3-1.7b-base", True),
        ("baseline@abc123", True),
        ("unsloth/Qwen3-1.7B (baseline, no fine-tuning)", True),
        ("rank32", False),
        ("minitaur-8b", False),
    ],
)
def test_baseline_detection(label: str, expected: bool) -> None:
    """Improvements are measured against the baseline, so it must be findable."""
    assert is_baseline(label) is expected


def test_improvement_is_measured_against_the_baseline() -> None:
    """0.92 to 0.64 is a 30% reduction, which is what the table must report."""
    rows = with_improvements([_row("baseline@abc", 0.92), _row("rank32", 0.644)])
    improvements = {row.label: row.improvement_pct for row in rows}
    assert improvements["rank32"] == pytest.approx(30.0)
    assert improvements["baseline@abc"] is None


def test_improvement_never_crosses_data_configurations() -> None:
    """A longer window improves the NLL by giving more context, not by training better.

    Crediting that to the fine-tuning is the easiest way to overstate a result here,
    so a run is only ever compared to the baseline of its own prepared data.
    """
    rows = with_improvements(
        [
            _row("baseline@short", 0.92, fingerprint="short"),
            _row("long_window", 0.50, fingerprint="long"),
        ]
    )
    assert {row.label: row.improvement_pct for row in rows}["long_window"] is None


def test_each_configuration_uses_its_own_baseline() -> None:
    """With a baseline per configuration, each run is scored against its own reference."""
    rows = with_improvements(
        [
            _row("baseline@a", 1.00, fingerprint="a"),
            _row("baseline@b", 0.80, fingerprint="b"),
            _row("run_a", 0.50, fingerprint="a"),
            _row("run_b", 0.40, fingerprint="b"),
        ]
    )
    improvements = {row.label: row.improvement_pct for row in rows}
    assert improvements["run_a"] == pytest.approx(50.0)
    assert improvements["run_b"] == pytest.approx(50.0)


def test_mixed_configurations_are_flagged_in_the_table() -> None:
    """Silently ranking incomparable rows together is the failure this guards against."""
    table = format_table([_row("a", 0.6, fingerprint="one"), _row("b", 0.7, fingerprint="two")])
    assert "WARNING" in table
    assert "not directly comparable" in table


def test_a_single_configuration_is_not_flagged() -> None:
    """A warning on every table would train the reader to ignore it."""
    table = format_table([_row("a", 0.6), _row("b", 0.7)])
    assert "WARNING" not in table


def test_references_appear_under_the_table() -> None:
    """The paper's numbers belong in view, labelled as the landmark they are."""
    table = format_table([_row("a", 0.6)])
    assert "0.44" in table
    assert "not a like-for-like comparison" in table


def test_empty_results_say_so() -> None:
    """An empty table must not render as a table with nothing in it."""
    assert "No results yet" in format_table([])


def test_the_fingerprint_is_read_from_the_key_evaluate_writes() -> None:
    """A key mismatch here fails open, not closed.

    Every row would come back with no fingerprint, land in the same bucket, and get
    compared to a baseline it was never comparable to -- with no error anywhere.
    """
    (row,) = load_rows({"run": {"nll": 0.6, "n_scored_tokens": 10, "data_fingerprint": "abc123"}})
    assert row.fingerprint == "abc123"


def test_progress_is_reported_once_the_floor_exists() -> None:
    """The table states each score as a share of a measured range.

    An absolute NLL invites comparison with numbers from other test sets, which is
    exactly the comparison this project cannot make.
    """
    table = format_table(
        [_row("model", 0.5388), _row("trivial-informed", 1.0198), _row("baseline", 0.9240)]
    )
    assert "Progress from the informed floor" in table
    assert "83.0%" in table  # (1.0198-0.5388)/(1.0198-0.44)


def test_no_progress_block_without_the_floor() -> None:
    """Reporting progress from a floor nobody measured would be an invented number."""
    assert "Progress from" not in format_table([_row("model", 0.5388)])


def test_the_trivial_rows_are_not_scored_against_themselves() -> None:
    """The floor is the origin of the scale, not a competitor on it."""
    table = format_table([_row("trivial-informed", 1.0198), _row("model", 0.5388)])
    block = table[table.index("Progress from") :]
    assert "trivial-informed" not in block
