"""Pipeline configuration, versioned in YAML and read by every stage.

Keeping the sequence length, the model name and the seed in a single place is not a
style preference. When the tokenizer that built the dataset and the model that trains
on it disagree about ``max_seq_length``, nothing raises: the run completes and the
reported metric is quietly wrong. One file that every stage reads removes that class
of failure entirely.

``extra="forbid"`` is deliberate too. A typo in a YAML key would otherwise be ignored
and the default silently used, which is the same failure wearing a different hat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ExperimentSelection(BaseModel):
    """Target experiments of the case study, grouped by cognitive domain.

    Values are matched as substrings against the ``experiment`` column of Psych-101,
    which is formatted as ``author+year/file.csv``. Substring matching lets a domain
    cover every file of the same study without listing them one by one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    risky_choice: tuple[str, ...] = ()
    categorization: tuple[str, ...] = ()
    reinforcement_learning: tuple[str, ...] = ()

    def all_patterns(self) -> tuple[str, ...]:
        """Return every pattern across all domains.

        Returns:
            Flat tuple of substrings to match against experiment names.
        """
        return self.risky_choice + self.categorization + self.reinforcement_learning

    def domain_of(self, experiment: str) -> str | None:
        """Return the domain an experiment belongs to.

        Args:
            experiment: Value of the ``experiment`` column.

        Returns:
            The domain name, or ``None`` when the experiment matches no pattern.
        """
        for domain, patterns in (
            ("risky_choice", self.risky_choice),
            ("categorization", self.categorization),
            ("reinforcement_learning", self.reinforcement_learning),
        ):
            if any(pattern in experiment for pattern in patterns):
                return domain
        return None


class DataConfig(BaseModel):
    """Where the data comes from and how it is cut into training examples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: str = "marcelbinz/Psych-101"
    dataset_split: str = "train"

    max_seq_length: int = Field(default=2048, ge=128)
    """Window size in tokens. The real ceiling on a 6GB card, far below the 32768 the
    Centaur authors use -- the gap is a declared limitation, not a bug."""

    window_stride: int | None = Field(default=None, ge=1)
    """Step between consecutive windows. ``None`` means no overlap (stride equals
    ``max_seq_length``). With overlap, tokens already scored in the previous window
    are masked out so no choice is counted twice."""

    val_fraction: float = Field(default=0.10, gt=0.0, lt=1.0)
    seed: int = 3407

    max_choices_per_domain: int | None = Field(default=None, ge=1)
    """Choice budget per cognitive domain. Without a cap the raw sizes decide the mix:
    ``peterson2021using`` alone carries 1,097,375 choices against 29,776 in
    ``badham2017deficits``, so training would be ~92% one experiment and the
    multi-domain specialisation claim would be untestable. ``None`` keeps everything,
    which is correct for evaluation but not for training."""

    catalog_sample_per_experiment: int = Field(default=30, ge=1)
    """Participants tokenized per experiment when measuring token-length statistics.
    Tokenizing all 60k transcripts to build a catalog would cost far more than the
    precision it buys."""

    @model_validator(mode="after")
    def _check_stride(self) -> Self:
        if self.window_stride is not None and self.window_stride > self.max_seq_length:
            msg = (
                f"window_stride ({self.window_stride}) exceeds max_seq_length "
                f"({self.max_seq_length}), which would skip tokens between windows"
            )
            raise ValueError(msg)
        return self

    @property
    def effective_stride(self) -> int:
        """Step actually used between windows.

        Returns:
            ``window_stride`` when set, otherwise ``max_seq_length`` (no overlap).
        """
        return self.window_stride or self.max_seq_length


class ModelConfig(BaseModel):
    """Base model and QLoRA adapter geometry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str = "unsloth/Qwen3-1.7B"
    tokenizer_name: str | None = None
    """Overrides the tokenizer source. Defaults to ``base_model``; only set this when
    evaluating an adapter whose tokenizer was saved separately."""

    load_in_4bit: bool = True

    lora_rank: int = Field(default=8, ge=1)
    """Rank 8 mirrors the Centaur paper and exists as the comparison baseline. Rank 8
    on a 1.7B model is far fewer trainable parameters than rank 8 on a 70B one, so
    higher ranks are a planned experiment, not a deviation."""

    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def tokenizer_source(self) -> str:
        """Model repository the tokenizer is loaded from.

        Returns:
            ``tokenizer_name`` when set, otherwise ``base_model``.
        """
        return self.tokenizer_name or self.base_model


class TrainingConfig(BaseModel):
    """Optimisation hyperparameters for the QLoRA run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    num_epochs: float = Field(default=1.0, gt=0.0)
    per_device_batch_size: int = Field(default=1, ge=1)
    grad_accumulation_steps: int = Field(default=32, ge=1)

    @property
    def effective_batch_size(self) -> int:
        """Batch size after gradient accumulation.

        Returns:
            Product of the per-device batch size and the accumulation steps.
        """
        return self.per_device_batch_size * self.grad_accumulation_steps


class PathsConfig(BaseModel):
    """Filesystem layout, relative to the repository root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    outputs: Path = Path("outputs")

    @property
    def splits(self) -> Path:
        """Path of the split manifest recording who is in train and validation.

        Returns:
            JSON path inside the processed directory.
        """
        return self.processed / "splits.json"

    @property
    def catalog(self) -> Path:
        """Path of the experiment catalog produced by the EDA stage.

        Returns:
            CSV path inside the interim directory.
        """
        return self.interim / "experiment_catalog.csv"


class PipelineConfig(BaseModel):
    """Full pipeline configuration. Every stage reads this and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiments: ExperimentSelection = ExperimentSelection()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathsConfig = PathsConfig()

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Location of the YAML file.

        Returns:
            The validated configuration.

        Raises:
            FileNotFoundError: When the file does not exist. Falling back to defaults
                would hide a wrong ``--config`` path, and a run that silently used the
                wrong hyperparameters is worse than one that refuses to start.
        """
        if not path.is_file():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
