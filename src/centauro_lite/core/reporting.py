"""Turning a pile of evaluation results into the table the thesis argues from.

An ablation table is only worth something if every row is comparable. Two rows measured
on different data configurations are two different questions with one column heading,
which is worse than not having the second row at all. So rows carry their data
fingerprint, and anything measured on a different one is separated out rather than
quietly ranked alongside.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from centauro_lite.core.floor import recovered_fraction
from centauro_lite.core.metrics import CENTAUR_NLL, COGNITIVE_MODELS_NLL, LLAMA_BASE_NLL

INFORMED_FLOOR_LABEL = "trivial-informed"
"""The predictor that knows a session's response options and guesses evenly among them.
It is the honest zero of this task: everything below it is knowledge about the person
rather than about the instructions."""

TRIVIAL_PREFIX = "trivial-"

BASELINE_MARKERS = ("baseline", "-base")
"""Substrings that identify an untuned run. The baseline is the reference every other
row's improvement is measured against, so it has to be findable."""


class ReportRow(BaseModel):
    """One evaluated model in the comparison table."""

    model_config = ConfigDict(frozen=True)

    label: str
    nll: float
    n_scored_tokens: int
    fingerprint: str | None = None
    """Evaluation fingerprint: what the row was measured *on*. Rows sharing it are
    comparable even when they trained on different amounts of data."""

    per_experiment: dict[str, float] = {}
    improvement_pct: float | None = None
    """Reduction in NLL against the baseline of the same data configuration. ``None``
    when no baseline was measured for that configuration -- in which case the row's
    absolute NLL is all that can honestly be said about it."""


def load_rows(results: Mapping[str, Any]) -> list[ReportRow]:
    """Read the accumulated results file into rows.

    Args:
        results: Parsed contents of ``eval_results.json``. Keys starting with an
            underscore are metadata rather than measurements.

    Returns:
        One row per evaluated model, unsorted.

    Note:
        The key is ``data_fingerprint``, matching what ``evaluate`` writes. Reading a
        key that is not there yields ``None`` for every row, which does not fail: it
        silently collapses every configuration into one bucket and disables the
        cross-configuration guard exactly when it is needed.
    """
    return [
        ReportRow(
            label=label,
            nll=float(entry["nll"]),
            n_scored_tokens=int(entry["n_scored_tokens"]),
            # Prefer the evaluation fingerprint: comparability is decided by what a
            # row was measured on, not by how its training data was tokenized. Older
            # results carry only the data fingerprint, and for those the two coincide.
            fingerprint=entry.get("eval_fingerprint") or entry.get("data_fingerprint"),
            per_experiment={k: float(v) for k, v in entry.get("per_experiment", {}).items()},
        )
        for label, entry in results.items()
        if not label.startswith("_")
    ]


def is_baseline(label: str) -> bool:
    """Whether a label names an untuned model.

    Args:
        label: The row's label.

    Returns:
        ``True`` when the label looks like a baseline.
    """
    lowered = label.lower()
    return any(marker in lowered for marker in BASELINE_MARKERS)


def with_improvements(rows: Sequence[ReportRow]) -> list[ReportRow]:
    """Attach each row's improvement over the baseline of its own configuration.

    Args:
        rows: Rows from :func:`load_rows`.

    Returns:
        The same rows with ``improvement_pct`` filled in where a matching baseline
        exists.

    Note:
        Matching is per fingerprint on purpose. Comparing a 4096-token run against a
        2048-token baseline would credit the extra context to the fine-tuning, which
        is the single easiest way to overstate a result in this project.
    """
    baselines = {row.fingerprint: row.nll for row in rows if is_baseline(row.label)}

    enriched: list[ReportRow] = []
    for row in rows:
        reference = baselines.get(row.fingerprint)
        improvement = (
            100.0 * (reference - row.nll) / reference
            if reference is not None and reference > 0 and not is_baseline(row.label)
            else None
        )
        enriched.append(row.model_copy(update={"improvement_pct": improvement}))
    return enriched


def rank(rows: Sequence[ReportRow]) -> list[ReportRow]:
    """Order rows best first.

    Args:
        rows: Rows to order.

    Returns:
        Rows sorted by ascending NLL, since lower means the model was less surprised
        by what the human actually did.
    """
    return sorted(rows, key=lambda row: row.nll)


def _progress_block(rows: Sequence[ReportRow]) -> list[str]:
    """Express each score as progress along a range whose ends are both measured.

    Args:
        rows: Ranked report rows.

    Returns:
        Lines for the table, or nothing when the informed floor has not been computed.

    Note:
        An absolute NLL invites comparison with numbers from other test sets, which is
        exactly the comparison this project cannot make. Anchoring instead to the
        informed floor -- knowing the response options and nothing else -- and to the
        published reference states the same result in terms both of whose ends were
        measured here.
    """
    floor = next((row.nll for row in rows if row.label == INFORMED_FLOOR_LABEL), None)
    scored = [row for row in rows if not row.label.startswith(TRIVIAL_PREFIX)]
    if floor is None or floor <= CENTAUR_NLL or not scored:
        return []

    lines = [
        "",
        f"Progress from the informed floor ({floor:.4f}, knowing the options and nothing",
        f"else) towards the Centaur reference ({CENTAUR_NLL:.2f}):",
    ]
    for row in scored:
        share = recovered_fraction(trivial=floor, model=row.nll, reference=CENTAUR_NLL)
        lines.append(f"  {row.label:<28}{share:>7.1f}%")
    return lines


def format_table(rows: Sequence[ReportRow]) -> str:
    """Render the comparison as fixed-width text.

    Args:
        rows: Rows to render, already ranked.

    Returns:
        The table, including the published reference values underneath and a warning
        when rows span more than one data configuration.
    """
    if not rows:
        return "No results yet. Run `evaluate` at least once."

    width = max(len(row.label) for row in rows) + 2
    lines = [f"{'model':<{width}}{'NLL':>9}{'vs base':>10}{'tokens':>12}  data", "-" * (width + 45)]
    for row in rows:
        improvement = f"{row.improvement_pct:+.1f}%" if row.improvement_pct is not None else "-"
        fingerprint = row.fingerprint or "unknown"
        lines.append(
            f"{row.label:<{width}}{row.nll:>9.4f}{improvement:>10}"
            f"{row.n_scored_tokens:>12,}  {fingerprint}"
        )

    fingerprints = {row.fingerprint for row in rows}
    if len(fingerprints) > 1:
        lines += [
            "",
            "WARNING: these rows span more than one data configuration. Rows with",
            "different fingerprints were measured on differently prepared data and",
            "are not directly comparable -- compare within a fingerprint.",
        ]

    lines += _progress_block(rows)
    lines += [
        "",
        "Published references (all 160 experiments, different tokenizer -- a landmark,",
        "not a like-for-like comparison):",
        f"  Centaur 70B                    {CENTAUR_NLL:.2f}",
        f"  Specialised cognitive models   {COGNITIVE_MODELS_NLL:.2f}",
        f"  Llama 3.1 70B, no fine-tuning  {LLAMA_BASE_NLL:.2f}",
    ]
    return "\n".join(lines)
