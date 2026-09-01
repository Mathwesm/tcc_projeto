"""Tests for the pipeline configuration.

The configuration exists to stop stages from disagreeing about ``max_seq_length`` and
friends. These tests check that it fails loudly when it cannot do that job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from centauro_lite.models.pipeline_config import (
    DataConfig,
    ExperimentSelection,
    PipelineConfig,
    TrainingConfig,
)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo in a YAML key must fail, not fall back to the default.

    Silently ignoring ``max_seq_lenght`` would run the whole pipeline at the default
    length while the author believes it was changed.
    """
    config_file = tmp_path / "typo.yaml"
    config_file.write_text("data:\n  max_seq_lenght: 512\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        PipelineConfig.from_yaml(config_file)


def test_missing_file_raises_instead_of_defaulting(tmp_path: Path) -> None:
    """A wrong ``--config`` path must stop the run rather than use defaults."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        PipelineConfig.from_yaml(tmp_path / "absent.yaml")


def test_empty_file_yields_defaults(tmp_path: Path) -> None:
    """An empty but present file is a valid way to ask for every default."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")
    assert PipelineConfig.from_yaml(config_file).data.max_seq_length == 2048


def test_stride_larger_than_window_is_rejected() -> None:
    """A stride beyond the window would skip tokens between windows.

    Those tokens are choices the metric would never see, which quietly shrinks the
    evaluation set.
    """
    with pytest.raises(ValidationError, match="window_stride"):
        DataConfig(max_seq_length=1024, window_stride=2048)


def test_stride_defaults_to_no_overlap() -> None:
    """Without an explicit stride, windows are contiguous and non-overlapping."""
    assert DataConfig(max_seq_length=1024).effective_stride == 1024


def test_overlapping_stride_is_allowed() -> None:
    """Overlap is a legitimate choice; only strides beyond the window are not."""
    assert DataConfig(max_seq_length=1024, window_stride=512).effective_stride == 512


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_val_fraction_must_leave_both_splits_populated(fraction: float) -> None:
    """A validation fraction of 0 or 1 leaves one split empty and the metric undefined."""
    with pytest.raises(ValidationError):
        DataConfig(val_fraction=fraction)


def test_effective_batch_size_matches_the_paper() -> None:
    """Batch 1 with 32 accumulation steps is the ~32 effective batch Centaur used."""
    assert TrainingConfig().effective_batch_size == 32


def test_domain_lookup_matches_by_substring() -> None:
    """Experiment names are ``author+year/file.csv``; a domain covers every file."""
    selection = ExperimentSelection(
        risky_choice=("peterson2021using",),
        categorization=("badham2017deficits",),
    )
    assert selection.domain_of("peterson2021using/exp1.csv") == "risky_choice"
    assert selection.domain_of("badham2017deficits/data.csv") == "categorization"
    assert selection.domain_of("someone2020other/exp.csv") is None


def test_all_patterns_covers_every_domain() -> None:
    """Filtering uses one flat list; a domain left out would drop silently."""
    selection = ExperimentSelection(
        risky_choice=("a",), categorization=("b",), reinforcement_learning=("c",)
    )
    assert set(selection.all_patterns()) == {"a", "b", "c"}


def test_config_is_immutable() -> None:
    """A stage mutating the shared config would desynchronise the stages after it."""
    config = PipelineConfig()
    with pytest.raises(ValidationError):
        config.data.max_seq_length = 512  # type: ignore[misc]


def test_fingerprint_changes_with_window_size() -> None:
    """Raising max_seq_length must invalidate the prepared data.

    Without this, the dataset on disk stays tokenized at the old length while the
    config says otherwise: training runs, evaluation runs, and the reported NLL
    describes windows of a size nobody asked for. Nothing else in the stack notices.
    """
    assert (
        PipelineConfig(data=DataConfig(max_seq_length=2048)).data_fingerprint
        != PipelineConfig(data=DataConfig(max_seq_length=4096)).data_fingerprint
    )


def test_fingerprint_changes_with_the_tokenizer() -> None:
    """Two tokenizers cut the same text differently, and the metric is per token."""
    from centauro_lite.models.pipeline_config import ModelConfig

    assert (
        PipelineConfig(model=ModelConfig(base_model="unsloth/Qwen3-1.7B")).data_fingerprint
        != PipelineConfig(model=ModelConfig(base_model="unsloth/Qwen3-0.6B")).data_fingerprint
    )


def test_fingerprint_changes_with_the_experiment_selection() -> None:
    """Adding an experiment changes what is measured, so results must not be pooled."""
    assert (
        PipelineConfig(experiments=ExperimentSelection(risky_choice=("a",))).data_fingerprint
        != PipelineConfig(experiments=ExperimentSelection(risky_choice=("a", "b"))).data_fingerprint
    )


def test_fingerprint_ignores_the_training_hyperparameters() -> None:
    """The point of a sweep is many training configs sharing one dataset.

    If the learning rate changed the fingerprint, every run would rebuild the data and
    no two rows would be comparable.
    """
    assert (
        PipelineConfig(training=TrainingConfig(learning_rate=5e-5)).data_fingerprint
        == PipelineConfig(training=TrainingConfig(learning_rate=2e-4)).data_fingerprint
    )


def test_fingerprint_ignores_the_catalog_sample_size() -> None:
    """That knob only affects the EDA statistics, never the prepared windows."""
    assert (
        PipelineConfig(data=DataConfig(catalog_sample_per_experiment=5)).data_fingerprint
        == PipelineConfig(data=DataConfig(catalog_sample_per_experiment=50)).data_fingerprint
    )


def test_every_sweep_config_shares_the_default_data_configuration() -> None:
    """A sweep row measured on different data is not an ablation, it is a second study.

    This reads the real files, so adding a config that quietly changes max_seq_length
    or the experiment list fails here rather than after hours of GPU time.
    """
    default = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    sweep_dir = Path("configs/sweep")
    configs = sorted(sweep_dir.glob("*.yaml"))
    assert configs, "the sweep directory is empty"
    for path in configs:
        assert PipelineConfig.from_yaml(path).data_fingerprint == default.data_fingerprint, path


def test_a_different_model_keeps_the_same_participants() -> None:
    """Evaluating a second model must not also move the held-out people.

    If the split changed at the same time as the tokenizer, the two results would
    differ for two reasons at once and neither could be attributed to the model.
    """
    from centauro_lite.models.pipeline_config import ModelConfig

    qwen = PipelineConfig(model=ModelConfig(base_model="unsloth/Qwen3-1.7B"))
    llama = PipelineConfig(model=ModelConfig(base_model="marcelbinz/Llama-3.1-Minitaur-8B"))
    assert qwen.split_fingerprint == llama.split_fingerprint
    assert qwen.data_fingerprint != llama.data_fingerprint


def test_a_different_window_keeps_the_same_participants() -> None:
    """Window size changes the tokenization, never who is held out."""
    short = PipelineConfig(data=DataConfig(max_seq_length=2048))
    long = PipelineConfig(data=DataConfig(max_seq_length=4096))
    assert short.split_fingerprint == long.split_fingerprint
    assert short.data_fingerprint != long.data_fingerprint


def test_changing_the_selection_moves_both_fingerprints() -> None:
    """Different participants means a different split and different windows."""
    one = PipelineConfig(experiments=ExperimentSelection(risky_choice=("a",)))
    two = PipelineConfig(experiments=ExperimentSelection(risky_choice=("a", "b")))
    assert one.split_fingerprint != two.split_fingerprint
    assert one.data_fingerprint != two.data_fingerprint


def test_the_subsampling_budget_moves_the_split() -> None:
    """A different choice budget selects different participants."""
    assert (
        PipelineConfig(data=DataConfig(max_choices_per_domain=30000)).split_fingerprint
        != PipelineConfig(data=DataConfig(max_choices_per_domain=10000)).split_fingerprint
    )


def test_datasets_of_different_tokenizations_do_not_collide() -> None:
    """Two tokenizations must land in different directories.

    A shared directory would mean the second `prepare` either overwrites the first
    model's token ids or is silently skipped, and one of the two models would then be
    evaluated on ids from the other's vocabulary.
    """
    from centauro_lite.models.pipeline_config import ModelConfig

    config = PipelineConfig()
    other = PipelineConfig(model=ModelConfig(base_model="marcelbinz/Llama-3.1-Minitaur-8B"))
    assert config.paths.dataset_dir(config.data_fingerprint) != other.paths.dataset_dir(
        other.data_fingerprint
    )


def test_the_minitaur_config_shares_the_split_and_not_the_tokenization() -> None:
    """The real config file, checked against the real default.

    This is the comparison the thesis rests on: same held-out participants, each model
    reading its own vocabulary.
    """
    default = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    minitaur = PipelineConfig.from_yaml(Path("configs/minitaur.yaml"))
    assert minitaur.split_fingerprint == default.split_fingerprint
    assert minitaur.data_fingerprint != default.data_fingerprint
