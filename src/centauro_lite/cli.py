"""Command line interface. Every pipeline stage is a subcommand here.

Stages are commands rather than numbered scripts so they all read the same
configuration object. Numbered scripts drift: each one grows its own copy of
``max_seq_length``, the copies disagree, and the run that follows is wrong without
raising anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
from centauro_lite.core.reporting import format_table, load_rows, rank, with_improvements
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

SweepDirOption = Annotated[
    Path,
    typer.Option("--configs", help="Directory of YAML files, one per sweep run."),
]

ForceOption = Annotated[
    bool,
    typer.Option("--force", help="Re-run configurations that already have a result."),
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

    manifest_path = config.paths.manifest_path(config.split_fingerprint)
    if reuse_splits and manifest_path.is_file():
        assignment = load_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
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
    dataset_dir = config.paths.dataset_dir(config.data_fingerprint)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(splits).save_to_disk(str(dataset_dir))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest(
                assignment,
                {
                    "split_fingerprint": config.split_fingerprint,
                    "data_fingerprint": config.data_fingerprint,
                    "tokenizer": config.model.tokenizer_source,
                    "stats": stats,
                    "config": config.model_dump(mode="json"),
                },
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Split manifest written to {}", manifest_path)
    (dataset_dir / "fingerprint.json").write_text(
        json.dumps(
            {
                "data_fingerprint": config.data_fingerprint,
                "split_fingerprint": config.split_fingerprint,
                "tokenizer": config.model.tokenizer_source,
                "max_seq_length": config.data.max_seq_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    typer.echo("")
    typer.echo(f"Dataset: {dataset_dir}")
    typer.echo(f"Manifest: {manifest_path}")
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
    _require_matching_data(config)
    dataset = load_from_disk(str(config.paths.dataset_dir(config.data_fingerprint)))
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
) -> None:
    """Measure the NLL over human choices on the validation split.

    Run this at least twice: once with no adapter for the baseline, once with the
    trained one. Without the baseline there is no way to claim the fine-tuning did
    anything at all.

    To evaluate a different model -- Minitaur-8B, say -- point ``--config`` at a
    configuration naming it, and prepare the data with that config first. There is
    deliberately no flag to swap the model without swapping the config: the prepared
    dataset stores token ids, and ids belong to one vocabulary. Qwen3 id 2610 is
    "You"; the same id under Llama's vocabulary is " askear". A model reading another
    model's ids sees noise, produces a plausible number, and raises nothing.

    Args:
        config_path: Location of the YAML configuration.
        adapter: Trained adapter to load on top of the base model.
        label: Name for this result in the report.
    """
    from datasets import load_from_disk

    from centauro_lite.services.modeling import load_model

    config = _bootstrap(config_path)
    _require_matching_data(config)
    dataset = load_from_disk(str(config.paths.dataset_dir(config.data_fingerprint)))["validation"]

    model, tokenizer = load_model(
        config, source=str(adapter) if adapter else None, for_inference=True
    )

    accumulator = _accumulate_nll(model, tokenizer, dataset, config)
    result = accumulator.result(label or _default_label(adapter, config))
    _write_results(config, result)
    _report(result)


def _default_label(adapter: Path | None, config: PipelineConfig) -> str:
    """Name a result so the report is still readable months later.

    Args:
        adapter: Adapter passed on the command line, if any.
        config: The pipeline configuration.

    Returns:
        A descriptive label.
    """
    if adapter:
        return f"{config.model.base_model} + {adapter.name}"
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
    results[result.label] = result.model_dump() | {"data_fingerprint": config.data_fingerprint}
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


def _require_matching_data(config: PipelineConfig) -> None:
    """Refuse to run when the prepared data does not match the configuration.

    Raising here is the whole point. Raise ``max_seq_length`` in a config and the
    dataset on disk is still tokenized at the old length: training would run,
    evaluation would run, and the reported NLL would describe windows of a size the
    configuration says they are not. Nothing else in the stack notices.

    Args:
        config: The pipeline configuration.

    Raises:
        RuntimeError: When no dataset was prepared, or when it was prepared from a
            different data configuration.
    """
    dataset_dir = config.paths.dataset_dir(config.data_fingerprint)
    stamp = dataset_dir / "fingerprint.json"
    if not stamp.is_file():
        msg = (
            f"No dataset prepared for this configuration at {dataset_dir}.\n"
            f"  tokenizer: {config.model.tokenizer_source}\n"
            f"  max_seq_length: {config.data.max_seq_length}\n"
            f"Run `prepare --config <this config>` first. A dataset prepared for a "
            f"different model cannot be reused: the stored token ids belong to that "
            f"model's vocabulary and mean something else under another one."
        )
        raise RuntimeError(msg)

    stored = json.loads(stamp.read_text(encoding="utf-8"))
    if stored.get("data_fingerprint") != config.data_fingerprint:
        msg = (
            f"The dataset at {dataset_dir} carries fingerprint "
            f"{stored.get('data_fingerprint')} but the config wants "
            f"{config.data_fingerprint}. Run `prepare` again."
        )
        raise RuntimeError(msg)


@app.command()
def sweep(
    config_dir: SweepDirOption = Path("configs/sweep"),
    base_config: ConfigOption = DEFAULT_CONFIG_PATH,
    force: ForceOption = False,
) -> None:
    """Train and evaluate every configuration in a directory, then report.

    Each configuration runs in its own subprocess. That is not ceremony: loading and
    discarding quantized models repeatedly in one process leaves VRAM fragmented, and
    on a 16GB card the fourth run is the one that fails. A fresh process per run also
    means one crash costs one row instead of the whole sweep.

    Completed runs are skipped, so a session that dies halfway resumes where it
    stopped rather than repeating hours of finished work.

    Args:
        config_dir: Directory of YAML files, one per experiment.
        base_config: Configuration used for the shared baseline measurement.
        force: Re-run configurations that already have a result.
    """
    # Recursive so `--configs configs/sweep` runs everything while
    # `--configs configs/sweep/group_a` runs one group. The groups exist because a
    # Kaggle batch session is capped at 12 hours and the full sweep is longer than
    # that; a run killed at the cap loses whatever it had not finished.
    configs = sorted(config_dir.rglob("*.yaml"))
    if not configs:
        typer.echo(f"No configurations found in {config_dir}")
        raise typer.Exit(code=1)

    config = _bootstrap(base_config)
    done = _completed_runs(config)
    logger.info("Sweep: {} configurations, {} already done", len(configs), len(done))

    _ensure_prepared(base_config, config)
    baseline_label = f"baseline@{config.data_fingerprint}"
    if baseline_label not in done or force:
        _run_stage(["evaluate", "--config", str(base_config), "--label", baseline_label])

    failures: list[str] = []
    for path in configs:
        run_id = path.stem
        if run_id in done and not force:
            logger.info("Skipping {} (already measured)", run_id)
            continue

        run_config = PipelineConfig.from_yaml(path)
        _ensure_prepared(path, run_config)

        adapter = config.paths.outputs / run_id / "adapter"
        logger.info("=== {} ===", run_id)
        if not _run_stage(["train", "--config", str(path), "--output", str(adapter)]):
            failures.append(run_id)
            continue
        if not _run_stage(
            ["evaluate", "--config", str(path), "--adapter", str(adapter), "--label", run_id]
        ):
            failures.append(run_id)

    if failures:
        typer.echo(f"\nFailed: {', '.join(failures)}")
    report(base_config)


def _completed_runs(config: PipelineConfig) -> set[str]:
    """Labels already present in the results file.

    Args:
        config: The pipeline configuration.

    Returns:
        The set of labels already measured.
    """
    path = config.paths.outputs / "eval_results.json"
    if not path.is_file():
        return set()
    stored = json.loads(path.read_text(encoding="utf-8"))
    return {key for key in stored if not key.startswith("_")}


def _ensure_prepared(config_path: Path, config: PipelineConfig) -> None:
    """Rebuild the dataset when the configuration asks for different data.

    Args:
        config_path: Path of the configuration file, passed on to ``prepare``.
        config: The parsed configuration.
    """
    stamp = config.paths.dataset_dir(config.data_fingerprint) / "fingerprint.json"
    if stamp.is_file():
        return
    stored = None
    logger.info("Data configuration changed ({} -> {}); preparing", stored, config.data_fingerprint)
    _run_stage(["prepare", "--config", str(config_path)])


def _run_stage(arguments: list[str]) -> bool:
    """Run one pipeline stage in a fresh subprocess.

    Args:
        arguments: CLI arguments after the module name.

    Returns:
        ``True`` when the stage exited cleanly.
    """
    command = [sys.executable, "-m", "centauro_lite", *arguments]
    logger.info("$ {}", " ".join(command))
    completed = subprocess.run(command, check=False)  # noqa: S603 - arguments are ours
    if completed.returncode != 0:
        logger.error("Stage failed with exit code {}", completed.returncode)
    return completed.returncode == 0


@app.command()
def report(config_path: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Render the comparison table and the charts from the accumulated results.

    Args:
        config_path: Location of the YAML configuration.
    """
    config = _bootstrap(config_path)
    path = config.paths.outputs / "eval_results.json"
    if not path.is_file():
        typer.echo(f"No results at {path}. Run `evaluate` first.")
        raise typer.Exit(code=1)

    rows = rank(with_improvements(load_rows(json.loads(path.read_text(encoding="utf-8")))))
    table = format_table(rows)
    typer.echo("")
    typer.echo(table)

    (config.paths.outputs / "report.txt").write_text(table + "\n", encoding="utf-8")
    _plot(rows, config.paths.outputs / "figures")


def _plot(rows: Any, directory: Path) -> None:
    """Write the ablation and per-experiment charts.

    Args:
        rows: Ranked report rows.
        directory: Where the figures go.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on a notebook runner or a CI machine
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(9, max(3, 0.45 * len(rows))))
    labels = [row.label for row in rows][::-1]
    values = [row.nll for row in rows][::-1]
    axes.barh(labels, values, color="#4C72B0")
    axes.axvline(CENTAUR_NLL, color="#C44E52", linestyle="--", label=f"Centaur 70B ({CENTAUR_NLL})")
    axes.set_xlabel("NLL over human choices (lower is better)")
    axes.set_title("Ablation: every run on the same held-out participants")
    axes.legend()
    figure.savefig(directory / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    experiments = sorted({name for row in rows for name in row.per_experiment})
    if not experiments:
        return
    figure, axes = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(rows))
    for index, row in enumerate(rows):
        positions = [position + index * width for position in range(len(experiments))]
        axes.bar(
            positions,
            [row.per_experiment.get(name, 0.0) for name in experiments],
            width=width,
            label=row.label,
        )
    axes.set_xticks([position + 0.4 for position in range(len(experiments))])
    axes.set_xticklabels(experiments, rotation=20, ha="right")
    axes.set_ylabel("NLL")
    axes.set_title("Per experiment: the average hides where a model actually wins")
    axes.legend(fontsize="small")
    figure.savefig(directory / "per_experiment.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Figures written to {}", directory)


def main() -> int:
    """Run the CLI.

    Returns:
        ``0`` on success. Typer raises :class:`SystemExit` on failure, so a scheduler
        still sees a non-zero code.
    """
    app()
    return 0
