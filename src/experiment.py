"""Orchestrate resumable continual-learning experiments and persist all evidence."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from .config import ABLATIONS, Config, DreamMode
from .data import TaskData, build_task_stream, stream_manifest
from .dreaming import (
    DreamBank, collect_sources, map_classes_to_dreams, refresh_inventory,
)
from .evaluation import compute_metrics, evaluate_task
from .models import ConditionalVAE, build_model
from .replay import ReservoirBuffer
from .training import train_phase, update_buffer
from .visualization import dream_preview, plot_accuracy_matrix, plot_dream_preview


@dataclass
class ExperimentResults:
    """Expose final metrics, histories, model state, mappings, and saved run path."""
    accuracy_matrix: np.ndarray
    metrics: dict[str, float]
    class_to_slot: dict[int, int]
    training_history: dict[str, list[dict]]
    dream_history: dict[str, list[dict]]
    dream_previews: dict[str, dict]
    model: ConditionalVAE
    run_dir: Path
    completed_tasks: int
    transfer: dict[str, np.ndarray]
    events: list[dict]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that priority order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rng_state() -> dict:
    """Capture every active random generator used by the experiment."""
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict) -> None:
    """Restore a previously captured experiment random state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def _loader_states(tasks: list[TaskData]) -> list[dict]:
    """Capture explicit generator states for every task loader."""
    result = []
    for task in tasks:
        state = {}
        for name in ("train_loader", "mapping_loader", "test_loader", "validation_loader"):
            generator = getattr(task, name).generator
            if generator is None:
                raise RuntimeError("Task loaders require explicit reproducible generators.")
            state[name] = generator.get_state()
        result.append(state)
    return result


def _load_checkpoint(path: Path) -> dict:
    """Load one trusted local checkpoint containing Python objects."""
    # Checkpoints include Python/NumPy RNG state: only load your own trusted runs.
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_json(path: Path, value: dict) -> None:
    """Atomically write strict JSON without non-finite numeric literals."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, state: dict) -> None:
    """Atomically persist a Torch task-boundary checkpoint."""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _config_from_dict(values: dict) -> Config:
    """Reconstruct a typed configuration from serialized values."""
    return Config(**{
        **values, "data_root": Path(values["data_root"]),
        "output_dir": Path(values["output_dir"]),
    })


def _load_base_model(
    path: Path, cfg: Config, model: ConditionalVAE,
) -> dict[str, object]:
    """Validate and load a selected task-zero model checkpoint."""
    state = _load_checkpoint(path)
    if "model" not in state or "protocol" not in state:
        raise ValueError("Base-model checkpoint must contain model and protocol metadata.")
    protocol = state["protocol"]
    if not isinstance(protocol, dict) or "config" not in protocol:
        raise ValueError("Base-model checkpoint is missing its training configuration.")
    trained_cfg = _config_from_dict(protocol["config"])
    required = (
        "seed", "base_classes", "classes_per_task", "num_tasks", "image_size",
        "model_variant", "latent_dim", "condition_dim",
    )
    mismatches = [
        name for name in required if getattr(trained_cfg, name) != getattr(cfg, name)
    ]
    if mismatches:
        raise ValueError(
            "Base-model checkpoint is incompatible for: " + ", ".join(mismatches)
        )
    model.load_state_dict(state["model"])
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": state.get("epoch"),
        "metrics": state.get("metrics"),
        "protocol": protocol,
    }


def _results(state: dict, model: ConditionalVAE, run_dir: Path) -> ExperimentResults:
    """Convert internal checkpoint state into the public result container."""
    return ExperimentResults(
        accuracy_matrix=state["accuracy_matrix"], metrics=state["metrics"],
        class_to_slot=state["class_to_slot"], training_history=state["training_history"],
        dream_history=state["dream_history"], dream_previews=state["dream_previews"],
        model=model, run_dir=run_dir, completed_tasks=state["next_task"],
        transfer=state["transfer"], events=state["events"],
    )


def load_run(run_dir: Path, *, device: torch.device | None = None) -> ExperimentResults:
    """Load a completed or interrupted trusted run for analysis.

    Args:
        run_dir: Directory containing the run's `checkpoint.pt`.
        device: Optional destination device, or the best available device.

    Returns:
        Reconstructed experiment results and model.
    """
    state = _load_checkpoint(run_dir / "checkpoint.pt")
    cfg = _config_from_dict(state["config"])
    model = build_model(cfg).to(device or default_device())
    model.load_state_dict(state["model"])
    model.eval()
    return _results(state, model, run_dir)


def run_experiment(
    cfg: Config, *, mode: DreamMode = "directed_with_reuse",
    device: torch.device | None = None, visualize: bool = False,
    verbose: bool = True, resume_from: Path | None = None,
    base_model_checkpoint: Path | None = None,
) -> ExperimentResults:
    """Run or resume one complete continual-learning ablation.

    Args:
        cfg: Fixed data, model, optimization, replay, and dream settings.
        mode: Replay or dream ablation to execute.
        device: Optional training device, or the best available device.
        visualize: Whether to display dream and accuracy diagnostics.
        verbose: Whether to print phase and task progress.
        resume_from: Existing run directory to resume at a task boundary.
        base_model_checkpoint: Optional selected task-zero model used to skip
            redundant base training.

    Returns:
        Completed metrics, histories, model, mappings, and artifact locations.
    """
    if mode not in ABLATIONS:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {ABLATIONS}.")
    seed_everything(cfg.seed)
    device = device or default_device()
    tasks = build_task_stream(cfg)
    manifest = stream_manifest(tasks, cfg.data_root)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    model = build_model(cfg).to(device)
    base_model = None
    if resume_from is None and base_model_checkpoint is not None:
        base_model = _load_base_model(base_model_checkpoint, cfg, model)
    # Construct the measured random reference without perturbing learner RNG.
    rng = _rng_state()
    seed_everything(cfg.seed + 10_000)
    reference = build_model(cfg).to(device).eval()
    _restore_rng(rng)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    buffer = ReservoirBuffer(cfg.replay_size, cfg.seed)
    bank = None
    if resume_from is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = cfg.output_dir / mode / f"seed-{cfg.seed}" / f"{timestamp}-{uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", {"config": cfg.to_dict(), "mode": mode})
        _write_json(run_dir / "data_manifest.json", manifest)
        state = {
            "config": cfg.to_dict(), "mode": mode, "next_task": 0,
            "manifest_hash": fingerprint,
            "accuracy_matrix": np.full((cfg.num_tasks, cfg.num_tasks), np.nan),
            "transfer": {
                key: np.full(cfg.num_tasks, np.nan)
                for key in ("pretrain_full", "random_full", "pretrain_restricted", "random_restricted")
            },
            "class_to_slot": {}, "training_history": {}, "dream_history": {},
            "dream_previews": {}, "events": [], "metrics": {},
            "base_model": base_model,
        }
    else:
        run_dir = resume_from
        state = _load_checkpoint(run_dir / "checkpoint.pt")
        if _config_from_dict(state["config"]).to_dict() != cfg.to_dict() or state["mode"] != mode:
            raise ValueError("Resume requires the original configuration and mode.")
        if state["manifest_hash"] != fingerprint:
            raise ValueError("Dataset partition changed since the checkpoint.")
        if state["device_type"] != device.type:
            raise ValueError("Exact resumption requires the same device type.")
        if base_model_checkpoint is not None:
            digest = hashlib.sha256(base_model_checkpoint.read_bytes()).hexdigest()
            if (state.get("base_model") or {}).get("sha256") != digest:
                raise ValueError("Resume base-model checkpoint does not match the original.")
        model.load_state_dict(state["model"])
        reference.load_state_dict(state["reference"])
        optimizer.load_state_dict(state["optimizer"])
        buffer = ReservoirBuffer.from_state_dict(state["buffer"])
        if state["bank"] is not None:
            bank = DreamBank.from_snapshot(state["bank"], device)
        for task, saved in zip(tasks, state["loader_states"], strict=True):
            for name, value in saved.items():
                generator = getattr(task, name).generator
                if generator is None:
                    raise RuntimeError("Missing loader generator during resumption.")
                generator.set_state(value.cpu())
        _restore_rng(state["rng"])
    if verbose:
        print(f"[setup] {mode}, device={device}, run={run_dir}", flush=True)
    mapping = state["class_to_slot"]
    for task_id in range(state["next_task"], cfg.num_tasks):
        task = tasks[task_id]
        if task_id == 0:
            mapping.update(dict(zip(task.class_ids, range(cfg.base_classes), strict=True)))
        else:
            available = (
                bank.slots if bank is not None
                else [slot for slot in range(cfg.total_classes) if slot not in mapping.values()]
            )
            reuse = mode in ("directed_with_reuse", "random_dreams")
            if reuse:
                incoming, scores = map_classes_to_dreams(
                    model, task.mapping_loader, list(task.class_ids), available, device
                )
                reference_mapping, _ = map_classes_to_dreams(
                    reference, task.mapping_loader, list(task.class_ids), available, device
                )
            else:
                incoming = dict(zip(task.class_ids, sorted(available)[:len(task.class_ids)], strict=True))
                reference_mapping, scores = incoming, None
            state["events"].append({
                "task": task_id, "event": "incoming_mapping", "mapping": dict(incoming),
                "reference_mapping": dict(reference_mapping),
                "scores": scores.tolist() if scores is not None else None,
            })
            transfer = state["transfer"]
            for name, classifier, assigned in (
                ("pretrain", model, incoming), ("random", reference, reference_mapping),
            ):
                transfer[f"{name}_full"][task_id] = evaluate_task(
                    classifier, task.test_loader, assigned, list(range(cfg.total_classes)), device
                )
                transfer[f"{name}_restricted"][task_id] = evaluate_task(
                    classifier, task.test_loader, assigned, sorted(assigned.values()), device
                )
            mapping.update(incoming)
            if bank is not None:
                bank = bank.retire(list(incoming.values()))
        stage = f"task_{task_id}/real"
        if task_id == 0 and state.get("base_model") is not None:
            state["training_history"][stage] = []
            state["events"].append({
                "task": 0, "event": "base_model_loaded",
                "checkpoint": state["base_model"],
            })
        else:
            state["training_history"][stage] = train_phase(
                model, task.train_loader, bank, mapping, buffer, optimizer, cfg, device,
                epochs=cfg.base_epochs if task_id == 0 else cfg.incremental_epochs,
                stage=stage, verbose=verbose,
            )
        update_buffer(buffer, task.train_loader)
        if task_id < cfg.num_tasks - 1:
            if mode != "replay_only":
                available = [
                    slot for slot in range(cfg.total_classes) if slot not in mapping.values()
                ]
                # Generation uses only this task and the bounded real replay inventory.
                sources, source_labels = collect_sources(
                    model, task.mapping_loader, buffer, cfg, device
                )
                bank, trajectory = refresh_inventory(
                    model, bank, list(task.class_ids), available, sources,
                    source_labels, mapping, cfg, device, random_dreams=mode == "random_dreams",
                )
                state["dream_history"][f"task_{task_id}"] = trajectory
                state["events"].append({
                    "task": task_id, "event": "dream_refresh", "slots": list(bank.slots),
                    "guides": list(bank.guides), "observed_classes": sorted(mapping),
                    "source_labels": sorted(set(source_labels.cpu().tolist())),
                })
            # Every condition gets the same additional real/replay training budget.
            stage = f"task_{task_id}/dream_training"
            state["training_history"][stage] = train_phase(
                model, task.train_loader, bank, mapping, buffer, optimizer, cfg, device,
                epochs=cfg.dream_epochs, stage=stage, verbose=verbose,
            )
            state["events"].append({
                "task": task_id, "event": "dream_slots_trained",
                "slots": list(bank.slots) if bank is not None else [],
            })
            if bank is not None:
                sources, source_labels = collect_sources(
                    model, task.mapping_loader, buffer, cfg, device
                )
                preview = dream_preview(model, bank, sources, source_labels)
                state["dream_previews"][f"task_{task_id}"] = preview
                if visualize:
                    plot_dream_preview(preview)
        for previous in range(task_id + 1):
            task_mapping = {label: mapping[label] for label in tasks[previous].class_ids}
            state["accuracy_matrix"][task_id, previous] = evaluate_task(
                model, tasks[previous].test_loader, task_mapping, sorted(mapping.values()), device
            )
        state["next_task"] = task_id + 1
        if state["next_task"] == cfg.num_tasks:
            transfer = state["transfer"]
            state["metrics"] = compute_metrics(
                state["accuracy_matrix"], transfer["pretrain_full"], transfer["random_full"],
                transfer["pretrain_restricted"], transfer["random_restricted"],
                [len(item.test_loader.dataset) for item in tasks],
            )
        state.update({
            "model": model.state_dict(), "reference": reference.state_dict(),
            "optimizer": optimizer.state_dict(), "buffer": buffer.state_dict(),
            "bank": bank.snapshot() if bank is not None else None,
            "rng": _rng_state(), "loader_states": _loader_states(tasks),
            "device_type": device.type,
        })
        _save_checkpoint(run_dir / f"task_{task_id:02d}.pt", state)
        _save_checkpoint(run_dir / "checkpoint.pt", state)
        _write_json(run_dir / "history.json", {
            "training": state["training_history"], "dreams": state["dream_history"],
            "events": state["events"], "completed_tasks": state["next_task"],
        })
        np.save(run_dir / "accuracy_matrix.npy", state["accuracy_matrix"])
        if verbose:
            print(f"[task {task_id}] seen accuracy={state['accuracy_matrix'][task_id, :task_id + 1]}", flush=True)
    _write_json(run_dir / "metrics.json", state["metrics"])
    if visualize:
        plot_accuracy_matrix(state["accuracy_matrix"], state["metrics"])
    model.eval()
    return _results(state, model, run_dir)
