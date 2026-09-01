"""Command line interface. Every pipeline stage is a subcommand here.

Stages are commands rather than numbered scripts so they all read the same
configuration object. Numbered scripts drift: each one grows its own copy of
``max_seq_length``, the copies disagree, and the run that follows is wrong without
raising anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from centauro_lite.config import settings
from centauro_lite.core.catalog import build_catalog, catalog_totals, iter_transcripts
from centauro_lite.core.chunking import iter_windows
from centauro_lite.core.masking import count_choices
from centauro_lite.core.metrics import (
    CENTAUR_NLL,
    COGNITIVE_MODELS_NLL,
    LLAMA_BASE_NLL,
    NllAccumulator,
    NllResult,
    scored_token_count,
)
from centauro_lite.core.sampling import ParticipantRef, balance_domains, choices_per_domain
from centauro_lite.core.splits import (
    leaked_participants,
    load_manifest,
    manifest,
    split_participants,
)
from centauro_lite.models.pipeline_config import DEFAULT_CONFIG_PATH, PipelineConfig
from centauro_lite.utils.logger import setup_logging

app = typer.Typer(
    add_completion=False,
    help="Centauro-Lite: QLoRA fine-tuning of small models on Psych-101.",
)

FITS_THRESHOLD_PCT = 99.0
"""An experiment counts as fitting in one window when nearly every transcript does.
Demanding a literal 100% would let a single outlier participant misrepresent an
experiment that needs no chunking in practice."""

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the pipeline configuration YAML."),
]

ReuseSplitsOption = Annotated[
    bool,
    typer.Option(
        "--reuse-splits/--resample",
        help="Reuse the committed split manifest instead of sampling again.",
    ),
]

OutputOption = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="Where to write the trained adapter."),
]

AdapterOption = Annotated[
    Path | None,
    typer.Option("--adapter", "-a", help="Trained adapter; omit for the untuned baseline."),
]

LabelOption = Annotated[
    str | None,
    typer.Option("--label", "-l", help="Name for this result in the report."),
]

ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model", "-m", help="Evaluate another model, e.g. marcelbinz/Llama-3.1-Minitaur-8B."
    ),
]


@app.callback()
def _root() -> None:
    """Keep stage names as explicit subcommands.

    Typer collapses a single-command app into a bare command, which would make
    ``eda`` an unexpected argument today and silently rename the entrypoint once the
    second stage lands. Registering a callback pins the subcommand form from the
    start.
    """


def _bootstrap(config_path: Path) -> PipelineConfig:
    """Configure logging and load the pipeline configuration.

    Args:
        config_path: Location of the YAML configuration.

    Returns:
        The validated configuration.
    """
    setup_logging(
        log_dir=settings.log_dir,
        level=settings.log_level,
        serialize=settings.log_serialize,
    )
    config = PipelineConfig.from_yaml(config_path)
    logger.info("Loaded config from {}", config_path)
    return config


@app.command()
def eda(config_path: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Catalog every experiment in Psych-101 and write it to the interim directory.

    Run this before choosing the case-study experiments. The catalog reports, per
    experiment, how many participants and choices exist and what share of transcripts
    actually fits in one window -- the number that decides whether truncation is
    acceptable or whether the transcripts have to be chunked.

    Args:
        config_path: Location of the YAML configuration.
    """
    import pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = _bootstrap(config_path)

    logger.info("Loading {} (~859 MB on first run)", config.data.dataset_name)
    dataset = load_dataset(config.data.dataset_name, split=config.data.dataset_split)
    logger.info("Loaded {} rows", len(dataset))

    logger.info("Loading tokenizer {}", config.model.tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_source)

    logger.info(
        "Building catalog (sampling {} participants per experiment for token stats)",
        config.data.catalog_sample_per_experiment,
    )
    catalog = build_catalog(
        iter_transcripts(dataset),
        tokenizer,
        max_seq_length=config.data.max_seq_length,
        sample_per_experiment=config.data.catalog_sample_per_experiment,
        domain_of=config.experiments.domain_of,
    )

    output = config.paths.catalog
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([stats.model_dump() for stats in catalog])
    frame.to_csv(output, index=False, encoding="utf-8")
    logger.info("Catalog written to {}", output)

    totals = catalog_totals(catalog)
    logger.info(
        "Totals: {} experiments, {} participants, {} choices",
        totals["n_experiments"],
        totals["n_participants"],
        totals["n_choices"],
    )

    typer.echo("")
    typer.echo(f"Catalog: {output}")
    typer.echo(
        f"  {totals['n_experiments']} experiments · "
        f"{totals['n_participants']:,} participants · "
        f"{totals['n_choices']:,} choices"
    )
    typer.echo("  Published reference: 160 experiments · 60,092 participants · 10,681,650 choices")
    typer.echo("")

    fits = [stats for stats in catalog if stats.fits_in_window_pct >= FITS_THRESHOLD_PCT]
    typer.echo(
        f"  {len(fits)} of {len(catalog)} experiments fit entirely in "
        f"{config.data.max_seq_length} tokens; the rest need chunking."
    )

    tagged = [stats for stats in catalog if stats.domain is not None]
    if tagged:
        typer.echo("")
        typer.echo("Case-study experiments currently configured:")
        for stats in tagged:
            typer.echo(
                f"  [{stats.domain}] {stats.experiment} · "
                f"{stats.n_participants} participants · "
                f"{stats.n_choices_total:,} choices · "
                f"{stats.tokens_mean:,.0f} tokens avg · "
                f"{stats.windows_mean} windows/participant"
            )


@app.command()
def prepare(
    config_path: ConfigOption = DEFAULT_CONFIG_PATH,
    reuse_splits: ReuseSplitsOption = True,
) -> None:
    """Filter, balance, split and window the dataset into trainable examples.

    The order of operations is the whole point: filter, then balance, then split by
    participant, and only then cut into windows. Windowing before splitting would put
    slices of the same session on both sides, and validation would be measuring
    memorisation.

    The split is read back from the committed manifest by default rather than
    resampled. A seed alone does not pin a split: it pins a permutation of whatever
    row order the installed library version produced, and that order can change.
    Reading the manifest is what lets a run on Kaggle prove it evaluated the same
    held-out participants as the run on the laptop.

    Args:
        config_path: Location of the YAML configuration.
        reuse_splits: Read the manifest when it exists. Pass ``--resample`` to
            deliberately draw a new split, which invalidates comparisons against
            every result measured on the old one.
    """
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    config = _bootstrap(config_path)
    dataset = _load_target_rows(config)

    refs = [
        ParticipantRef(
            experiment=row.experiment,
            participant=row.participant,
            domain=config.experiments.domain_of(row.experiment) or "unknown",
            n_choices=count_choices(row.text),
        )
        for row in iter_transcripts(dataset)
    ]
    logger.info("Matched {} participants across the target experiments", len(refs))

    if reuse_splits and config.paths.splits.is_file():
        assignment = load_manifest(json.loads(config.paths.splits.read_text(encoding="utf-8")))
        selected = [ref for ref in refs if (ref.experiment, ref.participant) in assignment.keys]
        logger.info(
            "Reusing the committed split manifest: {} participants",
            len(assignment.keys),
        )
        if len(selected) != len(assignment.keys):
            msg = (
                f"The manifest names {len(assignment.keys)} participants but only "
                f"{len(selected)} were found in the dataset. The manifest and the "
                f"configured experiments disagree; rerun with --resample or fix the config."
            )
            raise RuntimeError(msg)
    else:
        selected = balance_domains(
            refs,
            max_choices_per_domain=config.data.max_choices_per_domain,
            seed=config.data.seed,
        )
        assignment = split_participants(
            [(ref.experiment, ref.participant) for ref in selected],
            val_fraction=config.data.val_fraction,
            seed=config.data.seed,
        )
    logger.info(
        "{} participants selected, choices per domain: {}",
        len(selected),
        choices_per_domain(selected),
    )
    leaked = leaked_participants(assignment)
    if leaked:  # pragma: no cover - structurally impossible, kept as a tripwire
        msg = f"{len(leaked)} participants appear in both splits: {sorted(leaked)[:5]}"
        raise RuntimeError(msg)
    logger.info(
        "Split: {} train / {} validation participants",
        len(assignment.train),
        len(assignment.validation),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_source)
    texts = {
        (row.experiment, row.participant): row.text
        for row in iter_transcripts(dataset)
        if (row.experiment, row.participant) in assignment.keys
    }

    splits: dict[str, Dataset] = {}
    stats: dict[str, dict[str, int]] = {}
    for name, keys in (("train", assignment.train), ("validation", assignment.validation)):
        windows = _window_all(keys, texts, tokenizer, config)
        splits[name] = Dataset.from_list(windows)
        stats[name] = {
            "participants": len(keys),
            "windows": len(windows),
            "scored_tokens": sum(int(window["n_scored"]) for window in windows),
        }
        logger.info("{}: {}", name, stats[name])

    config.paths.processed.mkdir(parents=True, exist_ok=True)
    DatasetDict(splits).save_to_disk(str(config.paths.processed / "dataset"))
    config.paths.splits.write_text(
        json.dumps(
            manifest(assignment, {"stats": stats, "config": config.model_dump(mode="json")}),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Split manifest written to {}", config.paths.splits)

    typer.echo("")
    typer.echo(f"Dataset: {config.paths.processed / 'dataset'}")
    typer.echo(f"Manifest: {config.paths.splits}")
    for name, values in stats.items():
        typer.echo(
            f"  {name:<11} {values['participants']:>5} participants · "
            f"{values['windows']:>6} windows · {values['scored_tokens']:>8,} scored tokens"
        )
    typer.echo("")
    typer.echo("Choices per domain after balancing:")
    for domain, total in sorted(choices_per_domain(selected).items()):
        typer.echo(f"  {domain:<24} {total:>9,}")


def _load_target_rows(config: PipelineConfig) -> Any:
    """Load Psych-101 and keep only the rows of the configured experiments.

    Args:
        config: The pipeline configuration.

    Returns:
        The filtered dataset.

    Raises:
        typer.Exit: When no row matches, which almost always means a pattern in the
            config does not exist in the catalog.
    """
    from datasets import load_dataset

    patterns = config.experiments.all_patterns()
    if not patterns:
        typer.echo("No experiments configured. Fill `experiments` in the config first.")
        raise typer.Exit(code=1)

    logger.info("Loading {}", config.data.dataset_name)
    dataset = load_dataset(config.data.dataset_name, split=config.data.dataset_split)
    filtered = dataset.filter(lambda row: any(pattern in row["experiment"] for pattern in patterns))
    logger.info("Kept {} of {} rows", len(filtered), len(dataset))
    if len(filtered) == 0:
        typer.echo(f"No rows matched {patterns}. Check the names against the catalog.")
        raise typer.Exit(code=1)
    return filtered


def _window_all(
    keys: tuple[tuple[str, str], ...],
    texts: dict[tuple[str, str], str],
    tokenizer: Any,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Cut every selected participant into windows.

    Args:
        keys: Participants on this side of the split.
        texts: Transcript per participant key.
        tokenizer: Fast tokenizer providing offset mappings.
        config: The pipeline configuration.

    Returns:
        Window records ready to become a dataset.
    """
    records: list[dict[str, Any]] = []
    for experiment, participant in keys:
        for window in iter_windows(
            texts[(experiment, participant)],
            tokenizer,
            max_seq_length=config.data.max_seq_length,
            stride=config.data.effective_stride,
        ):
            records.append(
                {
                    "input_ids": window.input_ids,
                    "attention_mask": window.attention_mask,
                    "labels": window.labels,
                    "experiment": experiment,
                    "participant": participant,
                    "n_scored": window.n_scored,
                }
            )
    return records


@app.command()
def train(
    config_path: ConfigOption = DEFAULT_CONFIG_PATH,
    output: OutputOption = None,
) -> None:
    """Fine-tune the base model with QLoRA on the prepared dataset.

    Args:
        config_path: Location of the YAML configuration.
        output: Adapter destination. Defaults to ``<outputs>/adapter``. On a hosted
            notebook, point this at mounted storage: the runtime's own disk is thrown
            away when the session ends, and so is an unsaved run.
    """
    from datasets import load_from_disk

    from centauro_lite.services.modeling import attach_adapters, build_trainer, load_model

    config = _bootstrap(config_path)
    dataset = load_from_disk(str(config.paths.processed / "dataset"))
    logger.info(
        "Loaded {} train and {} validation windows",
        len(dataset["train"]),
        len(dataset["validation"]),
    )

    model, tokenizer = load_model(config)
    model = attach_adapters(model, config)

    destination = output or (config.paths.outputs / "adapter")
    destination.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(model, tokenizer, dataset, config, destination / "checkpoints")

    logger.info(
        "Training: lr={}, effective batch={}, epochs={}",
        config.training.learning_rate,
        config.training.effective_batch_size,
        config.training.num_epochs,
    )
    trainer.train()

    model.save_pretrained(str(destination))
    tokenizer.save_pretrained(str(destination))
    logger.info("Adapter saved to {}", destination)
    typer.echo("")
    typer.echo(f"Adapter: {destination}")
    typer.echo(f"Next: python -m centauro_lite evaluate --adapter {destination}")


@app.command()
def evaluate(
    config_path: ConfigOption = DEFAULT_CONFIG_PATH,
    adapter: AdapterOption = None,
    label: LabelOption = None,
    model_name: ModelOption = None,
) -> None:
    """Measure the NLL over human choices on the validation split.

    Run this at least twice: once with no adapter for the baseline, once with the
    trained one. Without the baseline there is no way to claim the fine-tuning did
    anything at all.

    The same command evaluates Minitaur-8B through ``--model``, and that is the
    comparison carrying the thesis: same code, same split, same metric. The paper's
    0.44 was measured on all 160 experiments with a different tokenizer, so it is a
    landmark rather than a like-for-like number.

    Args:
        config_path: Location of the YAML configuration.
        adapter: Trained adapter to load on top of the base model.
        label: Name for this result in the report.
        model_name: Evaluate a different model entirely.
    """
    from datasets import load_from_disk

    from centauro_lite.services.modeling import load_model

    config = _bootstrap(config_path)
    dataset = load_from_disk(str(config.paths.processed / "dataset"))["validation"]

    source = str(adapter) if adapter else model_name
    model, tokenizer = load_model(config, source=source, for_inference=True)

    accumulator = _accumulate_nll(model, tokenizer, dataset, config)
    result = accumulator.result(label or _default_label(adapter, model_name, config))
    _write_results(config, result)
    _report(result)


def _default_label(adapter: Path | None, model_name: str | None, config: PipelineConfig) -> str:
    """Name a result so the report is still readable months later.

    Args:
        adapter: Adapter passed on the command line, if any.
        model_name: Alternative model, if any.
        config: The pipeline configuration.

    Returns:
        A descriptive label.
    """
    if adapter:
        return f"{config.model.base_model} + {adapter.name}"
    if model_name:
        return model_name
    return f"{config.model.base_model} (baseline, no fine-tuning)"


def _accumulate_nll(
    model: Any, tokenizer: Any, dataset: Any, config: PipelineConfig
) -> NllAccumulator:
    """Run the model over the validation split, one experiment at a time.

    Evaluating per experiment rather than in one pass is what makes the per-experiment
    NLL available. The aggregate alone hides a model that is excellent at one task and
    useless at another, which is the most interesting thing a small specialised model
    could reveal.

    Args:
        model: The loaded model.
        tokenizer: Its tokenizer.
        dataset: The validation split.
        config: The pipeline configuration.

    Returns:
        The populated accumulator.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    from centauro_lite.services.modeling import TRAINING_COLUMNS

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    accumulator = NllAccumulator()
    model.eval()

    for experiment in sorted(set(dataset["experiment"])):
        subset = dataset.filter(lambda row, name=experiment: row["experiment"] == name)
        loader = DataLoader(
            subset.select_columns(list(TRAINING_COLUMNS)),
            batch_size=config.training.per_device_batch_size,
            collate_fn=collator,
        )
        logger.info("Evaluating {} ({} windows)", experiment, len(subset))
        for batch in loader:
            placed = {key: value.to(model.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**placed)
            accumulator.add(
                experiment,
                float(outputs.loss.item()),
                scored_token_count(batch["labels"].tolist()),
            )

    return accumulator


def _write_results(config: PipelineConfig, result: NllResult) -> None:
    """Append a result to the shared report file.

    Accumulating in one file rather than overwriting is what lets the baseline, the
    fine-tuned model and Minitaur sit side by side at the end.

    Args:
        config: The pipeline configuration.
        result: The finished evaluation.
    """
    path = config.paths.outputs / "eval_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    results[result.label] = result.model_dump()
    results["_references"] = result.comparison()
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to {}", path)


def _report(result: NllResult) -> None:
    """Print the result next to the published reference values.

    Args:
        result: The finished evaluation.
    """
    typer.echo("")
    typer.echo(
        f"NLL - {result.label}: {result.nll:.4f}   " f"({result.n_scored_tokens:,} scored tokens)"
    )
    typer.echo("")
    typer.echo("Per experiment:")
    for experiment, value in sorted(result.per_experiment.items()):
        tokens = result.per_experiment_tokens[experiment]
        typer.echo(f"  {experiment:<36} {value:.4f}  ({tokens:,} tokens)")
    typer.echo("")
    typer.echo("Published references, measured on all 160 experiments with a different")
    typer.echo("tokenizer - a landmark, not a like-for-like comparison:")
    typer.echo(f"  Centaur 70B                    {CENTAUR_NLL:.2f}")
    typer.echo(f"  Specialised cognitive models   {COGNITIVE_MODELS_NLL:.2f}")
    typer.echo(f"  Llama 3.1 70B, no fine-tuning  {LLAMA_BASE_NLL:.2f}")


def main() -> int:
    """Run the CLI.

    Returns:
        ``0`` on success. Typer raises :class:`SystemExit` on failure, so a scheduler
        still sees a non-zero code.
    """
    app()
    return 0
