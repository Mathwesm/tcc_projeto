"""Loading and training the model with unsloth.

This module is the only place that touches the GPU stack, and it is deliberately thin.
Everything with real logic -- masking, chunking, splitting, the NLL arithmetic -- lives
in ``core`` where it runs and is tested on a laptop without a GPU. What is left here is
plumbing, so a change in the unsloth API breaks one small file rather than the project.

Imports are function-local on purpose: ``torch`` and ``unsloth`` take tens of seconds to
import and are absent on the machine where the data stages run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from centauro_lite.models.pipeline_config import PipelineConfig

TRAINING_COLUMNS = ("input_ids", "attention_mask", "labels")
"""Columns the trainer accepts. ``experiment`` and ``participant`` ride along in the
dataset for per-experiment reporting, but a collator would choke on them."""


def _fast_model_class() -> Any:
    """Return whichever unsloth entrypoint this installed version provides.

    Unsloth renamed its loader from ``FastLanguageModel`` to the unified ``FastModel``,
    and Kaggle and Colab images pin different versions. Choosing at runtime beats
    pinning a version that the notebook environment may override anyway.

    Returns:
        The loader class.

    Raises:
        ImportError: When neither name exists, which means the install is broken
            rather than merely a different version.
    """
    import unsloth

    for name in ("FastModel", "FastLanguageModel"):
        candidate = getattr(unsloth, name, None)
        if candidate is not None:
            logger.info("Using unsloth.{}", name)
            return candidate

    msg = "Neither unsloth.FastModel nor unsloth.FastLanguageModel is available"
    raise ImportError(msg)


def load_model(
    config: PipelineConfig,
    *,
    source: str | None = None,
    for_inference: bool = False,
) -> tuple[Any, Any]:
    """Load the base model, or an adapter on top of it, in 4-bit.

    Args:
        config: The pipeline configuration.
        source: Model or adapter to load. Defaults to the configured base model.
        for_inference: Switch the model into unsloth's inference mode. Leave ``False``
            for training.

    Returns:
        The model and its tokenizer.

    Note:
        Baseline and fine-tuned model are both loaded in 4-bit. Comparing a bf16
        baseline against a 4-bit fine-tune would blend two effects -- quantization and
        training -- into one number, and the paper's question is about the second.
    """
    fast_model = _fast_model_class()
    model_source = source or config.model.base_model

    logger.info("Loading {} (4-bit={})", model_source, config.model.load_in_4bit)
    model, tokenizer = fast_model.from_pretrained(
        model_name=model_source,
        max_seq_length=config.data.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )

    if for_inference:
        fast_model.for_inference(model)
    return model, tokenizer


def attach_adapters(model: Any, config: PipelineConfig) -> Any:
    """Add the LoRA adapters that training will update.

    Args:
        model: A model returned by :func:`load_model`.
        config: The pipeline configuration.

    Returns:
        The model with adapters attached.
    """
    fast_model = _fast_model_class()
    logger.info(
        "Attaching LoRA adapters (rank={}, alpha={}, modules={})",
        config.model.lora_rank,
        config.model.lora_alpha,
        len(config.model.target_modules),
    )
    return fast_model.get_peft_model(
        model,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=list(config.model.target_modules),
        bias="none",
        # Not optional on a small card: without it the activations of a 2048-token
        # window do not fit alongside the optimizer state.
        use_gradient_checkpointing="unsloth",
        random_state=config.data.seed,
    )


def build_trainer(
    model: Any, tokenizer: Any, dataset: Any, config: PipelineConfig, output_dir: Path
) -> Any:
    """Assemble the Hugging Face trainer.

    Args:
        model: Model with adapters attached.
        tokenizer: Its tokenizer, used by the collator for padding.
        dataset: A ``DatasetDict`` with ``train`` and ``validation`` splits.
        config: The pipeline configuration.
        output_dir: Where checkpoints are written. On a hosted notebook this should
            live on mounted storage, because the session's own disk is discarded when
            the runtime is recycled.

    Returns:
        The configured trainer.
    """
    import torch
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    columns = list(TRAINING_COLUMNS)
    train_split = dataset["train"].select_columns(columns)
    eval_split = dataset["validation"].select_columns(columns)

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_batch_size,
        per_device_eval_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.grad_accumulation_steps,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        # Ampere and newer support bf16, which is far less prone than fp16 to the
        # loss quietly turning into NaN partway through a LoRA run.
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=config.data.seed,
        # The validation loss of an aggressive configuration bottoms out and then climbs
        # again -- rank 64 with lr 2e-4 reached its minimum at epoch 3 and was 11% worse
        # by epoch 5. Keeping the last epoch throws that minimum away.
        load_best_model_at_end=config.training.select_best_epoch,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks: list[TrainerCallback] = []
    if config.training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        )

    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train_split,
        eval_dataset=eval_split,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
        callbacks=callbacks,
    )
