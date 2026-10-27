"""Define validated experiment configurations and supported ablation modes."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

DreamMode = Literal[
    "replay_only", "random_dreams", "directed_no_reuse", "directed_with_reuse"
]
ModelVariant = Literal["legacy", "balanced"]
ABLATIONS: tuple[DreamMode, ...] = (
    "replay_only", "random_dreams", "directed_no_reuse", "directed_with_reuse"
)


@dataclass(frozen=True)
class Config:
    """Store every data, model, optimization, dreaming, and evaluation setting."""
    data_root: Path = Path("data")
    output_dir: Path = Path("outputs")
    seed: int = 42
    base_classes: int = 50
    classes_per_task: int = 5
    num_tasks: int = 11
    image_size: int = 84
    model_variant: ModelVariant = "legacy"
    latent_dim: int = 64
    condition_dim: int = 16
    batch_size: int = 32
    num_workers: int = 0
    base_epochs: int = 10
    incremental_epochs: int = 10
    dream_epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    replay_size: int = 2_000
    replay_weight: float = 1.0
    reconstruction_weight: float = 1.0
    kl_weight: float = 1e-3
    dream_weight: float = 1.0
    dream_steps: int = 200
    dream_learning_rate: float = 0.02
    dream_patience: int = 20
    dream_source_samples: int = 256
    dream_batch_size: int = 32
    offset_weight: float = 1e-2
    max_target_confidence: float = 0.8
    max_offset_rms: float = 1.5
    min_diversity_ratio: float = 0.4
    csv_test_samples_per_class: int = 100
    validation_samples_per_class: int = 50
    train_samples_per_class: int | None = None
    eval_samples_per_class: int | None = None

    def __post_init__(self) -> None:
        if self.model_variant not in ("legacy", "balanced"):
            raise ValueError("model_variant must be 'legacy' or 'balanced'.")
        positive = (
            "base_classes", "classes_per_task", "num_tasks", "image_size",
            "latent_dim", "condition_dim", "batch_size", "base_epochs",
            "incremental_epochs", "dream_epochs", "dream_steps",
            "dream_patience", "dream_source_samples", "dream_batch_size",
            "csv_test_samples_per_class",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.base_classes < 2 or self.num_tasks < 2 or self.image_size < 16 or self.latent_dim < 2:
            raise ValueError("Require >=2 base classes/tasks/latent dimensions and image_size >=16.")
        if self.replay_size < 0 or self.num_workers < 0 or self.validation_samples_per_class < 0:
            raise ValueError("Replay, workers and validation counts cannot be negative.")
        for name in ("train_samples_per_class", "eval_samples_per_class"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None.")
        for name in (
            "learning_rate", "dream_learning_rate", "max_offset_rms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "weight_decay", "replay_weight", "reconstruction_weight",
            "kl_weight", "dream_weight", "offset_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if not 0 < self.max_target_confidence < 1:
            raise ValueError("max_target_confidence must be between 0 and 1.")
        if not 0 < self.min_diversity_ratio <= 1:
            raise ValueError("min_diversity_ratio must be in (0, 1].")
        if self.dream_batch_size < 2 or self.dream_source_samples < 2:
            raise ValueError("Dream diversity requires at least two source samples.")

    @property
    def total_classes(self) -> int:
        """Return the fixed classifier capacity implied by the task schedule."""
        return self.base_classes + self.classes_per_task * (self.num_tasks - 1)

    def to_dict(self) -> dict:
        """Return a JSON-compatible dictionary containing every setting."""
        return {
            **asdict(self),
            "data_root": str(self.data_root),
            "output_dir": str(self.output_dir),
        }


def smoke_config(data_root: Path = Path("data"), output_dir: Path = Path("outputs")) -> Config:
    """Return a tiny deterministic configuration for fast pipeline validation.

    Args:
        data_root: Directory containing the input dataset.
        output_dir: Directory in which smoke-run artifacts are saved.

    Returns:
        A three-task configuration with small images, batches, and epoch counts.
    """
    return Config(
        data_root=data_root, output_dir=output_dir, base_classes=2,
        classes_per_task=1, num_tasks=3, image_size=32, latent_dim=8,
        condition_dim=4, batch_size=4, base_epochs=1, incremental_epochs=1,
        dream_epochs=1, dream_steps=3, dream_patience=2,
        dream_source_samples=16, dream_batch_size=4, replay_size=12,
        train_samples_per_class=4, eval_samples_per_class=4,
    )
