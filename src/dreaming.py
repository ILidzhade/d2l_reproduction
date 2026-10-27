"""Create, optimize, assign, retire, and persist latent dream inventories."""

from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
import random
from typing import Generator

import networkx as nx
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .config import Config
from .data import ImageDataset
from .models import ConditionalVAE
from .replay import ReservoirBuffer


@contextmanager
def frozen(model: nn.Module) -> Generator[None, None, None]:
    """Temporarily freeze parameters and evaluation mode, then restore both."""
    flags = [parameter.requires_grad for parameter in model.parameters()]
    training = model.training
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
        model.train(training)


class DreamBank(nn.Module):
    """Store one additive latent offset and observed guide for each dream slot."""
    def __init__(
        self, slots: list[int], guides: list[int], latent_dim: int, *, seed: int = 0,
    ) -> None:
        super().__init__()
        if len(slots) != len(guides) or len(slots) != len(set(slots)):
            raise ValueError("Dream slots must be unique and have one observed guide each.")
        self.slots = list(slots)
        self.guides = list(guides)
        self.offsets = nn.Parameter(0.02 * torch.randn(
            len(slots), latent_dim, generator=torch.Generator().manual_seed(seed)
        ))

    def __len__(self) -> int:
        return len(self.slots)

    def transform(self, z: Tensor, *, detach_offsets: bool = True) -> Tensor:
        """Apply every stored offset to every source latent."""
        offsets = self.offsets.detach() if detach_offsets else self.offsets
        return z.unsqueeze(0) + offsets.unsqueeze(1)

    def retire(self, slots: list[int]) -> "DreamBank":
        """Return a new dream bank without the supplied occupied slots."""
        keep = [i for i, slot in enumerate(self.slots) if slot not in slots]
        bank = DreamBank(
            [self.slots[i] for i in keep],
            [self.guides[i] for i in keep], self.offsets.shape[1],
        ).to(self.offsets.device)
        with torch.no_grad():
            bank.offsets.copy_(self.offsets[keep])
        return bank

    def snapshot(self) -> dict:
        """Return a CPU snapshot suitable for checkpoint persistence."""
        return {
            "slots": self.slots, "guides": self.guides,
            "offsets": self.offsets.detach().cpu().clone(),
        }

    @classmethod
    def from_snapshot(cls, state: dict, device: torch.device) -> "DreamBank":
        """Restore a dream bank snapshot on the requested device."""
        bank = cls(state["slots"], state["guides"], state["offsets"].shape[1]).to(device)
        with torch.no_grad():
            bank.offsets.copy_(state["offsets"])
        return bank


def _balanced_indices(
    labels: Sequence[int], count: int, rng: random.Random, *, exclude: int | None = None,
) -> list[int]:
    """Sample indices in randomized class-balanced rounds with optional exclusion."""
    if count <= 0:
        raise ValueError("Dream sampling requires a positive sample count.")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label != exclude:
            groups.setdefault(label, []).append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    selected = []
    while groups and len(selected) < count:
        classes = sorted(groups)
        rng.shuffle(classes)
        for label in classes:
            selected.append(groups[label].pop())
            if not groups[label]:
                del groups[label]
            if len(selected) == count:
                break
    return selected


def _dataset_labels(dataset: Dataset) -> list[int]:
    """Extract ordered integer labels from an image dataset or subset."""
    if isinstance(dataset, Subset):
        labels = _dataset_labels(dataset.dataset)
        return [labels[index] for index in dataset.indices]
    if isinstance(dataset, ImageDataset):
        return dataset.targets
    raise TypeError("Dream collection requires ImageDataset or its task Subset.")


@torch.no_grad()
def collect_sources(
    model: ConditionalVAE, loader: DataLoader, buffer: ReservoirBuffer,
    cfg: Config, device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect balanced posterior means and labels from current data and replay.

    Args:
        model: Conditional VAE used to encode images.
        loader: Current task's deterministic mapping loader.
        buffer: Bounded replay memory of observed real images.
        cfg: Source budget, batch size, worker count, and seed.
        device: Device used for encoding.

    Returns:
        Source latent means and their dataset labels on the requested device.
    """
    model.eval()
    features, labels = [], []
    rng = random.Random(cfg.seed)
    budget = cfg.dream_source_samples // (2 if len(buffer) else 1)
    indices = _balanced_indices(_dataset_labels(loader.dataset), budget, rng)
    source_loader = DataLoader(
        Subset(loader.dataset, indices), batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=loader.pin_memory,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    for images, batch_labels in source_loader:
        features.append(model.encode(images.to(device)).cpu())
        labels.append(batch_labels.cpu())
    if len(buffer):
        replay_indices = _balanced_indices(
            buffer.labels, cfg.dream_source_samples - len(indices), rng
        )
        images = torch.stack([buffer.images[index] for index in replay_indices])
        batch_labels = torch.tensor([buffer.labels[index] for index in replay_indices])
        features.append(model.encode(images.to(device)).cpu())
        labels.append(batch_labels)
    if not features:
        raise ValueError("Cannot generate dreams from an empty training stream.")
    return torch.cat(features).to(device), torch.cat(labels).to(device)


def assign_scores(
    scores: np.ndarray, class_ids: list[int], slots: list[int],
) -> dict[int, int]:
    """Find the maximum-weight one-to-one mapping from classes to output slots."""
    if scores.shape != (len(class_ids), len(slots)) or len(class_ids) > len(slots):
        raise ValueError("Assignment scores must cover every class and available slot.")
    if not np.isfinite(scores).all() or len(set(slots)) != len(slots):
        raise ValueError("Assignment requires finite scores and unique slots.")
    graph = nx.Graph()
    for row in range(len(class_ids)):
        for column in range(len(slots)):
            graph.add_edge(("class", row), ("slot", column), weight=float(scores[row, column]))
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    mapping = {}
    for left, right in matching:
        row, column = (left[1], right[1]) if left[0] == "class" else (right[1], left[1])
        mapping[class_ids[row]] = slots[column]
    if len(mapping) != len(class_ids):
        raise RuntimeError("Incomplete one-to-one dream assignment.")
    return mapping


@torch.no_grad()
def map_classes_to_dreams(
    model: ConditionalVAE, loader: DataLoader, class_ids: list[int],
    slots: list[int], device: torch.device,
) -> tuple[dict[int, int], np.ndarray]:
    """Map incoming classes to available slots by mean classifier probability.

    Args:
        model: Current latent classifier.
        loader: Incoming task's training-only mapping loader.
        class_ids: Incoming dataset labels to assign.
        slots: Available classifier outputs.
        device: Device used for inference.

    Returns:
        The optimal class-to-slot mapping and its mean probability matrix.
    """
    model.eval()
    scores = torch.zeros(len(class_ids), len(slots))
    counts = torch.zeros(len(class_ids))
    for images, labels in loader:
        probabilities = model(images.to(device)).softmax(-1)[:, slots].cpu()
        for row, label in enumerate(class_ids):
            mask = labels == label
            scores[row] += probabilities[mask].sum(0)
            counts[row] += int(mask.sum())
    if (counts == 0).any():
        raise ValueError("Mapping requires training examples for every incoming class.")
    means = (scores / counts[:, None]).numpy()
    return assign_scores(means, class_ids, slots), means


def optimize_dreams(
    model: ConditionalVAE, bank: DreamBank, sources: Tensor, source_labels: Tensor,
    class_to_slot: dict[int, int], cfg: Config,
) -> list[dict]:
    """Optimize class-directed offsets and retain the safest lowest-loss states.

    Args:
        model: Frozen latent classifier defining the dream objective.
        bank: Dream inventory whose offsets are optimized in place.
        sources: Observed current/replay latent means.
        source_labels: Dataset labels corresponding to source latents.
        class_to_slot: Observed real-class output assignments.
        cfg: Dream optimizer and stopping settings.

    Returns:
        Per-step optimization diagnostics for every observed guide.
    """
    history: list[dict] = []
    if not set(bank.guides) <= class_to_slot.keys():
        raise ValueError("Dream guides must all be observed real classes.")
    label_values = source_labels.detach().cpu().tolist()
    with frozen(model):
        for index, guide in enumerate(bank.guides):
            selected = _balanced_indices(
                label_values, cfg.dream_batch_size, random.Random(cfg.seed + guide),
                exclude=guide,
            )
            source = sources[selected].detach()
            label_counts = dict(Counter(label_values[i] for i in selected))
            if len(source) < 2:
                raise ValueError(
                    f"Guide {guide} needs >=2 observed training/replay sources from other classes."
                )
            offset = nn.Parameter(bank.offsets[index].detach().clone())
            optimizer = torch.optim.Adam([offset], lr=cfg.dream_learning_rate)
            targets = torch.full(
                (len(source),), class_to_slot[guide], device=source.device, dtype=torch.long
            )
            reference_diversity = F.normalize(source, dim=-1).var(0, unbiased=False).mean()
            if float(reference_diversity) <= 1e-12:
                raise ValueError("Source latent diversity is zero; cannot assess collapse.")
            best = offset.detach().clone()
            best_loss = float("inf")
            stale = 0
            reason = "step_limit"
            for step in range(cfg.dream_steps + 1):
                dreamed = source + offset
                logits = model.classify_features(dreamed)
                ce = F.cross_entropy(logits, targets)
                loss = ce + cfg.offset_weight * offset.square().mean()
                confidence = logits.softmax(-1)[:, class_to_slot[guide]].mean()
                diversity = (
                    F.normalize(dreamed, dim=-1).var(0, unbiased=False).mean()
                    / reference_diversity
                )
                rms = offset.square().mean().sqrt()
                values = torch.stack((loss, confidence, diversity, rms)).detach()
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError(f"Nonfinite dream optimization for guide {guide}.")
                row = {
                    "guide": guide, "slot": bank.slots[index], "step": step,
                    "loss": float(loss.detach()), "target_ce": float(ce.detach()),
                    "confidence": float(confidence.detach()),
                    "diversity_ratio": float(diversity.detach()),
                    "offset_rms": float(rms.detach()),
                    "source_labels": sorted(label_counts),
                    "source_label_counts": label_counts,
                }
                history.append(row)
                if row["confidence"] >= cfg.max_target_confidence:
                    reason = "confidence_limit"
                elif row["offset_rms"] > cfg.max_offset_rms:
                    reason = "displacement_limit"
                elif row["diversity_ratio"] < cfg.min_diversity_ratio:
                    reason = "diversity_limit"
                else:
                    if row["loss"] < best_loss - 1e-6:
                        best, best_loss, stale = offset.detach().clone(), row["loss"], 0
                    else:
                        stale += 1
                    if stale >= cfg.dream_patience:
                        reason = "plateau"
                    elif step < cfg.dream_steps:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        continue
                break
            if best_loss == float("inf"):
                raise RuntimeError(
                    f"No safe dream candidate for observed guide {guide}: {reason}. "
                    "Inspect the observed-data diagnostics before adjusting stopping thresholds."
                )
            with torch.no_grad():
                bank.offsets[index].copy_(best)
            history[-1]["stop_reason"] = reason
            history[-1]["retained_loss"] = best_loss
    return history


def refresh_inventory(
    model: ConditionalVAE, bank: DreamBank | None, guides: list[int],
    available: list[int], sources: Tensor, labels: Tensor,
    mapping: dict[int, int], cfg: Config, device: torch.device, *, random_dreams: bool,
) -> tuple[DreamBank, list[dict]]:
    """Create fresh dreams, assign slots, and merge them into the inventory.

    Args:
        model: Current latent classifier.
        bank: Existing persistent dream inventory, if any.
        guides: Newly observed classes used as dream targets.
        available: Classifier slots not occupied by real classes.
        sources: Current/replay source latent means.
        labels: Dataset labels for source latents.
        mapping: Observed real-class output assignments.
        cfg: Model dimensions and dream optimization settings.
        device: Device holding the model and dream bank.
        random_dreams: Whether to retain random offsets without optimization.

    Returns:
        The refreshed persistent bank and any directed optimization history.
    """
    guides = guides[:len(available)]
    if not guides:
        raise ValueError("Refreshing dreams requires guides and unused output slots.")
    fresh = DreamBank(
        available[:len(guides)], guides, cfg.latent_dim, seed=cfg.seed + len(mapping)
    ).to(device)
    history = [] if random_dreams else optimize_dreams(
        model, fresh, sources, labels, mapping, cfg
    )
    with torch.no_grad():
        dreamed = fresh.transform(sources)
        scores = model.classify_features(dreamed).softmax(-1)[..., available].mean(1).cpu().numpy()
    assignments = assign_scores(scores, list(range(len(guides))), available)
    fresh.slots = [assignments[i] for i in range(len(guides))]
    for row in history:
        row["assigned_slot"] = fresh.slots[fresh.guides.index(row["guide"])]
    if bank is None:
        return fresh, history
    retained = bank.retire(fresh.slots)
    merged = DreamBank(
        retained.slots + fresh.slots, retained.guides + fresh.guides, cfg.latent_dim
    ).to(device)
    with torch.no_grad():
        merged.offsets.copy_(torch.cat((retained.offsets, fresh.offsets)))
    return merged, history
