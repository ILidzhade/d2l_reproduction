"""Evaluate class-incremental accuracy, forgetting, and forward transfer."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import ConditionalVAE
from .training import remap_labels


@torch.no_grad()
def evaluate_task(
    model: ConditionalVAE, loader: DataLoader, mapping: dict[int, int],
    candidate_slots: list[int], device: torch.device,
) -> float:
    """Measure one task's accuracy over the supplied candidate output slots.

    Args:
        model: Trained conditional VAE and latent classifier.
        loader: Evaluation batches for one task.
        mapping: Dataset-label to classifier-slot assignments.
        candidate_slots: Output slots allowed during prediction.
        device: Device on which model inference is performed.

    Returns:
        The fraction of examples assigned to their mapped output slot.
    """
    if not candidate_slots or not set(mapping.values()) <= set(candidate_slots):
        raise ValueError("Evaluation candidates must contain all target slots.")
    model.eval()
    candidates = torch.tensor(candidate_slots, device=device)
    correct, total = 0, 0
    for images, labels in loader:
        logits = model(images.to(device))[:, candidate_slots]
        predictions = candidates[logits.argmax(-1)]
        targets = remap_labels(labels.to(device), mapping)
        correct += int((predictions == targets).sum())
        total += len(labels)
    if not total:
        raise ValueError("Cannot evaluate an empty loader.")
    return correct / total


def compute_metrics(
    accuracy_matrix: np.ndarray, pretrain_full: np.ndarray, random_full: np.ndarray,
    pretrain_restricted: np.ndarray, random_restricted: np.ndarray,
    task_sample_counts: list[int],
) -> dict[str, float]:
    """Derive final accuracy, forgetting, and measured-reference FWT metrics.

    Args:
        accuracy_matrix: Accuracy after each task on every task seen so far.
        pretrain_full: Full-head incoming-task accuracies before training.
        random_full: Full-head accuracies from the random reference model.
        pretrain_restricted: Task-restricted incoming accuracies before training.
        random_restricted: Task-restricted random-reference accuracies.
        task_sample_counts: Number of evaluation examples in each task.

    Returns:
        Final task-average and weighted accuracy, forgetting, and both FWT forms.
    """
    n = len(accuracy_matrix)
    if n < 2 or accuracy_matrix.shape != (n, n) or len(task_sample_counts) != n:
        raise ValueError("Metrics require a square >=2-task matrix and sample counts.")
    if any(count <= 0 for count in task_sample_counts):
        raise ValueError("Every task must have evaluation samples.")
    if not np.isfinite(accuracy_matrix[np.tril_indices(n)]).all():
        raise ValueError("Seen-task accuracy entries must be finite.")
    for values in (pretrain_full, random_full, pretrain_restricted, random_restricted):
        if values.shape != (n,) or not np.isfinite(values[1:]).all():
            raise ValueError("Every incoming task needs measured pretraining/reference accuracy.")
    final = accuracy_matrix[-1]
    forgetting = [
        np.max(accuracy_matrix[i:, i]) - final[i] for i in range(n - 1)
    ]
    full_fwt = float(np.mean(pretrain_full[1:] - random_full[1:]))
    return {
        "final_average_accuracy": float(np.mean(final)),
        "final_sample_weighted_accuracy": float(np.average(final, weights=task_sample_counts)),
        "average_forgetting": float(np.mean(forgetting)),
        "forward_transfer": full_fwt,
        "forward_transfer_full_head": full_fwt,
        "forward_transfer_task_restricted": float(
            np.mean(pretrain_restricted[1:] - random_restricted[1:])
        ),
    }
