"""Train the classifier and CVAE on current, replay, and dreamed latents."""

import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dreaming import DreamBank
from .models import ConditionalVAE, vae_losses
from .replay import ReservoirBuffer


def remap_labels(labels: Tensor, mapping: dict[int, int]) -> Tensor:
    """Convert dataset labels into their current classifier output slots."""
    missing = set(labels.detach().cpu().tolist()) - mapping.keys()
    if missing:
        raise ValueError(f"No observed output mapping for labels {sorted(missing)}.")
    return torch.tensor(
        [mapping[label] for label in labels.detach().cpu().tolist()],
        device=labels.device, dtype=torch.long,
    )


def train_phase(
    model: ConditionalVAE, loader: DataLoader, bank: DreamBank | None,
    mapping: dict[int, int], buffer: ReservoirBuffer,
    optimizer: torch.optim.Optimizer, cfg: Config, device: torch.device,
    *, epochs: int, stage: str, verbose: bool,
) -> list[dict]:
    """Train one real/replay/dream phase and return per-epoch loss summaries.

    Args:
        model: Conditional VAE and latent classifier to optimize.
        loader: Current task's shuffled real-image loader.
        bank: Optional dream inventory supplying auxiliary latent classes.
        mapping: Observed dataset-label to classifier-slot assignments.
        buffer: Bounded real replay memory.
        optimizer: Optimizer updating all trainable model parameters.
        cfg: Loss weights and batch settings.
        device: Device used for training.
        epochs: Number of complete passes through the current task.
        stage: Human-readable phase name used in diagnostics.
        verbose: Whether to print per-epoch summaries.

    Returns:
        One averaged loss record per completed epoch.
    """
    history = []
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        rows = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            mu, logvar = model.posterior(images)
            classification = F.cross_entropy(
                model.classify_features(mu), remap_labels(labels, mapping)
            )
            reconstruction, kl = vae_losses(model, images, labels, mu, logvar)
            total = classification + cfg.reconstruction_weight * reconstruction + cfg.kl_weight * kl
            source = mu
            replay_loss = total.new_zeros(())
            if len(buffer):
                replay_images, replay_labels = buffer.sample(cfg.batch_size)
                replay_images, replay_labels = replay_images.to(device), replay_labels.to(device)
                replay_mu, replay_logvar = model.posterior(replay_images)
                replay_ce = F.cross_entropy(
                    model.classify_features(replay_mu), remap_labels(replay_labels, mapping)
                )
                replay_rec, replay_kl = vae_losses(
                    model, replay_images, replay_labels, replay_mu, replay_logvar
                )
                replay_loss = (
                    replay_ce + cfg.reconstruction_weight * replay_rec + cfg.kl_weight * replay_kl
                )
                total = total + cfg.replay_weight * replay_loss
                source = torch.cat((source, replay_mu))
            dream_loss = total.new_zeros(())
            if bank is not None and len(bank):
                dreamed = bank.transform(source)
                targets = torch.tensor(
                    bank.slots, device=device, dtype=torch.long
                ).repeat_interleave(len(source))
                dream_loss = F.cross_entropy(
                    model.classify_features(dreamed.flatten(0, 1)), targets
                )
                total = total + cfg.dream_weight * dream_loss
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Nonfinite training loss in {stage}.")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            rows.append({
                "loss": float(total.detach()), "classification": float(classification.detach()),
                "reconstruction": float(reconstruction.detach()), "kl": float(kl.detach()),
                "replay": float(replay_loss.detach()), "dream": float(dream_loss.detach()),
            })
        if not rows:
            raise ValueError(f"Empty training loader in {stage}.")
        summary = {
            "epoch": epoch,
            **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "dream_slots": list(bank.slots) if bank is not None else [],
        }
        history.append(summary)
        if verbose:
            print(
                f"[{stage}] {epoch + 1}/{epochs} loss={summary['loss']:.4f} "
                f"CE={summary['classification']:.4f} rec={summary['reconstruction']:.4f} "
                f"KL={summary['kl']:.4f} dream={summary['dream']:.4f} "
                f"time={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return history


def update_buffer(buffer: ReservoirBuffer, loader: DataLoader) -> None:
    """Add every current-task training example to the reservoir exactly once."""
    for images, labels in loader:
        buffer.add_batch(images, labels)
