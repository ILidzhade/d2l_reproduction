"""Notebook/import facade; implementation lives in src/d2l.

The former deterministic autoencoder API has been replaced by a conditional VAE.
Use directed_with_reuse / directed_no_reuse instead of structured_* modes.
"""

from src import (
    ABLATIONS, Config, ConditionalVAE, DreamBank, DreamMode, ExperimentResults,
    ReservoirBuffer, TaskData, build_model, build_task_stream, compute_metrics,
    default_device, evaluate_task, load_run, map_classes_to_dreams, optimize_dreams,
    plot_accuracy_matrix, plot_dream_history, plot_dream_preview,
    plot_reconstructions, plot_task_stream, plot_training_history, run_experiment,
    seed_everything, show_task_samples, smoke_config,
)

__all__ = [
    "ABLATIONS", "Config", "ConditionalVAE", "DreamBank", "DreamMode",
    "ExperimentResults", "ReservoirBuffer", "TaskData", "build_model",
    "build_task_stream", "compute_metrics", "default_device", "evaluate_task",
    "load_run", "map_classes_to_dreams", "optimize_dreams", "plot_accuracy_matrix",
    "plot_dream_history", "plot_dream_preview", "plot_reconstructions",
    "plot_task_stream", "plot_training_history", "run_experiment",
    "seed_everything", "show_task_samples", "smoke_config",
]
