"""Reproduce the improved-CVAE Dream2Learn comparison and report figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import (
    Config, build_model, build_task_stream, default_device, load_run,
    run_experiment, seed_everything,
)
from src.replay import ReservoirBuffer
from src.training import train_phase


SEED = 42
BASE_EPOCHS = 100
MODES = ("replay_only", "random_dreams", "directed_with_reuse")
OUTPUT_DIR = Path("outputs")
BASE_CHECKPOINT = (
    OUTPUT_DIR / "phase7_experiment5_balanced_latent128_selected_seed-42.pt"
)
RESULT_OUTPUT = OUTPUT_DIR / "phase8_final_balanced_mechanism_seed-42.json"
FIGURE_OUTPUT = Path("docs/report/final-balanced-results.png")


def experiment_config() -> Config:
    """Return the fixed configuration used for every submitted result."""
    return Config(
        data_root=Path("data"), output_dir=OUTPUT_DIR, seed=SEED,
        base_classes=50, classes_per_task=5, num_tasks=11,
        image_size=84, model_variant="balanced", latent_dim=128,
        condition_dim=16, batch_size=32, num_workers=0,
        base_epochs=20, incremental_epochs=10, dream_epochs=2,
        learning_rate=3e-4, weight_decay=1e-4,
        replay_size=2_000, reconstruction_weight=1.0, kl_weight=1e-3,
        dream_weight=1.0, dream_steps=200, dream_learning_rate=0.02,
        validation_samples_per_class=50,
    )


def ensure_base_checkpoint(cfg: Config, device: torch.device) -> None:
    """Train and save the shared 100-epoch balanced-CVAE base when absent."""
    if BASE_CHECKPOINT.is_file():
        return
    seed_everything(cfg.seed)
    base_task = build_task_stream(cfg)[0]
    mapping = dict(zip(base_task.class_ids, range(cfg.base_classes), strict=True))
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_phase(
        model, base_task.train_loader, None, mapping, ReservoirBuffer(0),
        optimizer, cfg, device, epochs=BASE_EPOCHS, stage="base_selection",
        verbose=True,
    )
    BASE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": {
            "name": "balanced_latent128",
            "config": cfg.to_dict(),
            "max_epochs": BASE_EPOCHS,
            "objective": "classification_ce + 1.0*l1 + 0.001*mean_kl",
        },
        "epoch": BASE_EPOCHS,
        "model": model.state_dict(),
    }, BASE_CHECKPOINT)


def matching_run(cfg: Config, mode: str, base_sha256: str) -> Path | None:
    """Return the newest resumable run matching the complete fixed protocol."""
    root = cfg.output_dir / mode / f"seed-{cfg.seed}"
    if not root.exists():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        checkpoint = candidate / "checkpoint.pt"
        if not checkpoint.is_file():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base = state.get("base_model") or {}
        if (
            state.get("mode") == mode
            and state.get("config") == cfg.to_dict()
            and base.get("sha256") == base_sha256
        ):
            return candidate
    return None


def json_safe(value):
    """Convert arrays and non-finite floats into strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_results(cfg: Config, results: dict, base_sha256: str) -> None:
    """Write all condition metrics, matrices, transfer values, and paired effects."""
    conditions = {
        mode: {
            "run_dir": str(result.run_dir),
            "metrics": result.metrics,
            "accuracy_matrix": result.accuracy_matrix,
            "transfer": result.transfer,
        }
        for mode, result in results.items()
    }
    replay = conditions["replay_only"]["metrics"]
    for row in conditions.values():
        row["minus_replay"] = {
            key: row["metrics"][key] - replay[key] for key in replay
        }
    random_metrics = conditions["random_dreams"]["metrics"]
    directed_metrics = conditions["directed_with_reuse"]["metrics"]
    payload = {
        "protocol": {
            "seed": cfg.seed,
            "modes": list(MODES),
            "config": cfg.to_dict(),
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_sha256": base_sha256,
            "base_checkpoint_epoch": BASE_EPOCHS,
        },
        "conditions": conditions,
        "directed_minus_random": {
            key: directed_metrics[key] - random_metrics[key]
            for key in random_metrics
        },
    }
    RESULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(RESULT_OUTPUT)


def save_figure(results: dict) -> None:
    """Save the submitted final-accuracy and full-head-FWT comparison figure."""
    names = ("Replay", "Random dreams", "Directed dreams")
    colors = ("#777777", "#4c78a8", "#e45756")
    accuracy = [
        100 * results[mode].metrics["final_average_accuracy"] for mode in MODES
    ]
    fwt = [
        100 * results[mode].metrics["forward_transfer_full_head"] for mode in MODES
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, accuracy, color=colors)
    axes[0].set(
        ylabel="Final average accuracy (%)",
        title="Improved-CVAE final result",
    )
    axes[1].bar(names, fwt, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        ylabel="Full-head FWT (percentage points)",
        title="Paper-style forward transfer",
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run or resume the complete submitted experiment and regenerate its outputs."""
    cfg = experiment_config()
    device = default_device()
    ensure_base_checkpoint(cfg, device)
    base_sha256 = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    results = {}
    for mode in MODES:
        run_dir = matching_run(cfg, mode, base_sha256)
        if run_dir is None:
            result = run_experiment(
                cfg, mode=mode, device=device,
                base_model_checkpoint=BASE_CHECKPOINT,
            )
        else:
            state = torch.load(
                run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            result = (
                load_run(run_dir, device=device)
                if state["next_task"] == cfg.num_tasks
                else run_experiment(
                    cfg, mode=mode, device=device, resume_from=run_dir,
                    base_model_checkpoint=BASE_CHECKPOINT,
                )
            )
        results[mode] = result
    save_results(cfg, results, base_sha256)
    save_figure(results)
    print(f"Saved metrics to {RESULT_OUTPUT}")
    print(f"Saved figure to {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
