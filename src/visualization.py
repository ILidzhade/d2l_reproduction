"""Render task, reconstruction, dream, training, and evaluation diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .data import TaskData
from .dreaming import DreamBank
from .models import ConditionalVAE


def plot_task_stream(tasks: list[TaskData]) -> None:
    """Display the number of classes introduced by each task."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(tasks)), [len(task.class_ids) for task in tasks])
    ax.set(xlabel="Task", ylabel="Classes", title="Continual class order")
    fig.tight_layout()
    plt.show()


def show_task_samples(task: TaskData, max_images: int = 12) -> None:
    """Display a labelled grid of observed training examples from one task."""
    images, labels = next(iter(task.mapping_loader))
    grid = make_grid(images[:max_images], nrow=4)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_title(f"Observed training examples: {labels[:max_images].tolist()}")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def plot_reconstructions(
    model: ConditionalVAE, loader: DataLoader, device: torch.device, count: int = 6,
) -> None:
    """Display real images above their conditional posterior-mean reconstructions."""
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:count].to(device), labels[:count].to(device)
    decoded = model.reconstruct(images, labels)
    fig, axes = plt.subplots(2, len(images), figsize=(2 * len(images), 4), squeeze=False)
    for index in range(len(images)):
        axes[0, index].imshow(images[index].cpu().permute(1, 2, 0))
        axes[1, index].imshow(decoded[index].cpu().permute(1, 2, 0))
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    fig.suptitle("Real examples / conditional posterior-mean reconstructions")
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def dream_preview(
    model: ConditionalVAE, bank: DreamBank, sources: torch.Tensor,
    labels: torch.Tensor, count: int = 6,
) -> dict:
    """Decode representative dreams and return image, latent, and cycle diagnostics.

    Args:
        model: Conditional VAE used for decoding and re-encoding.
        bank: Current dream inventory.
        sources: Observed source latent means.
        labels: Dataset labels corresponding to the source latents.
        count: Maximum number of dream guides to preview.

    Returns:
        Decoded images, identities, latents, PCA coordinates, and cycle error.
    """
    model.eval()
    dreamed, guides, slots = [], [], []
    for index, guide in enumerate(bank.guides[:count]):
        selected = sources[labels != guide][:3]
        if not len(selected):
            raise ValueError(f"No non-guide preview sources for class {guide}.")
        dreamed.append(selected + bank.offsets[index])
        guides.extend([guide] * len(selected))
        slots.extend([bank.slots[index]] * len(selected))
    if not dreamed:
        raise ValueError("Cannot preview an empty dream inventory.")
    z = torch.cat(dreamed)
    conditions = torch.tensor(guides, device=z.device)
    decoded = model.decode(z, conditions)
    roundtrip = model.encode(decoded)
    centered = torch.cat((sources, z))
    centered = centered - centered.mean(0, keepdim=True)
    _, _, components = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    return {
        "images": decoded.cpu(), "guides": guides, "slots": slots,
        "latents": z.cpu(), "roundtrip_mse": float((roundtrip - z).square().mean()),
        "projection": projected.cpu(), "num_sources": len(sources),
    }


def plot_dream_preview(preview: dict) -> None:
    """Display decoded dreams beside their real/dream latent projection."""
    images = preview["images"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(make_grid(images, nrow=3).permute(1, 2, 0))
    axes[0].set_title("Individual dreams (3 sources per observed guide)")
    axes[0].axis("off")
    points = preview["projection"]
    n = preview["num_sources"]
    axes[1].scatter(points[:n, 0], points[:n, 1], s=10, alpha=0.3, label="real means")
    axes[1].scatter(points[n:, 0], points[n:, 1], marker="x", label="dreams")
    axes[1].set_title(f"Latent PCA; decode/encode MSE={preview['roundtrip_mse']:.4g}")
    axes[1].legend()
    fig.suptitle(f"Guide labels: {preview['guides']}; auxiliary slots: {preview['slots']}")
    fig.tight_layout()
    plt.show()


def plot_dream_history(history: list[dict]) -> None:
    """Plot dream loss, confidence, and diversity across optimization steps."""
    if not history:
        raise ValueError("No optimization trajectory (random/no-dream control or not run).")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for guide in sorted({row["guide"] for row in history}):
        rows = [row for row in history if row["guide"] == guide]
        for ax, key in zip(axes, ("target_ce", "confidence", "diversity_ratio"), strict=True):
            ax.plot([row["step"] for row in rows], [row[key] for row in rows], alpha=0.6)
            ax.set(xlabel="Optimization step", title=key)
    fig.tight_layout()
    plt.show()


def plot_training_history(history: dict[str, list[dict]]) -> None:
    """Plot total training loss for every real and dream phase."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for stage, rows in history.items():
        ax.plot([row["epoch"] for row in rows], [row["loss"] for row in rows], label=stage)
    ax.set(xlabel="Epoch within phase", ylabel="Loss", title="CVAE + replay + latent dreams")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    plt.show()


def plot_accuracy_matrix(matrix: np.ndarray, metrics: dict[str, float]) -> None:
    """Display the task accuracy matrix beside final continual-learning metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[0].set(xlabel="Evaluated task", ylabel="After task", title="Class-incremental accuracy")
    fig.colorbar(image, ax=axes[0])
    keys = (
        "final_average_accuracy", "final_sample_weighted_accuracy",
        "average_forgetting", "forward_transfer_full_head",
        "forward_transfer_task_restricted",
    )
    if metrics:
        axes[1].barh(keys, [metrics[key] for key in keys])
        axes[1].axvline(0, color="black", linewidth=0.5)
    else:
        axes[1].set_title("Incomplete run: final metrics unavailable")
    fig.tight_layout()
    plt.show()
