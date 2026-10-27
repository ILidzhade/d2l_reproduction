"""Leakage-controlled conditional-VAE latent adaptation of Dream2Learn."""

from .config import ABLATIONS, Config, DreamMode, ModelVariant, smoke_config
from .data import TaskData, build_task_stream
from .dreaming import DreamBank, map_classes_to_dreams, optimize_dreams
from .evaluation import compute_metrics, evaluate_task
from .experiment import (
    ExperimentResults, default_device, load_run, run_experiment, seed_everything,
)
from .models import BalancedConditionalVAE, ConditionalVAE, build_model
from .replay import ReservoirBuffer
from .visualization import (
    plot_accuracy_matrix, plot_dream_history, plot_dream_preview,
    plot_reconstructions, plot_task_stream, plot_training_history, show_task_samples,
)

__all__ = [
    "ABLATIONS", "Config", "DreamMode", "ModelVariant", "smoke_config", "TaskData",
    "build_task_stream", "DreamBank", "map_classes_to_dreams", "optimize_dreams",
    "compute_metrics", "evaluate_task", "ExperimentResults", "default_device",
    "load_run", "run_experiment", "seed_everything", "BalancedConditionalVAE",
    "ConditionalVAE", "build_model",
    "ReservoirBuffer", "plot_accuracy_matrix", "plot_dream_history",
    "plot_dream_preview", "plot_reconstructions", "plot_task_stream",
    "plot_training_history", "show_task_samples",
]
