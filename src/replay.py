"""Maintain a bounded real-image replay memory with reservoir sampling."""

import random

import torch
from torch import Tensor


class ReservoirBuffer:
    """Store an unbiased bounded sample of all observed real training images."""
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 0:
            raise ValueError("Replay capacity cannot be negative.")
        self.capacity = capacity
        self.seen = 0
        self.images: list[Tensor] = []
        self.labels: list[int] = []
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.images)

    def add_batch(self, images: Tensor, labels: Tensor) -> None:
        """Insert a labelled batch using standard reservoir replacement."""
        for image, label in zip(images.detach().cpu(), labels.detach().cpu(), strict=True):
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label))
            else:
                index = self._random.randrange(self.seen + 1)
                if index < self.capacity:
                    self.images[index] = image.clone()
                    self.labels[index] = int(label)
            self.seen += 1

    def sample(
        self, batch_size: int, *, rng: random.Random | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample up to the requested number of replay images and labels."""
        if not self.images or batch_size <= 0:
            raise ValueError("Sampling requires nonempty replay and a positive batch size.")
        indices = (rng or self._random).sample(range(len(self)), min(batch_size, len(self)))
        return (
            torch.stack([self.images[i] for i in indices]),
            torch.tensor([self.labels[i] for i in indices]),
        )

    def state_dict(self) -> dict:
        """Return all buffer contents and random state needed for exact resumption."""
        return {
            "capacity": self.capacity, "seen": self.seen, "images": self.images,
            "labels": self.labels, "random_state": self._random.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ReservoirBuffer":
        """Restore a replay buffer from a previously saved state dictionary."""
        buffer = cls(state["capacity"])
        buffer.seen = state["seen"]
        buffer.images = state["images"]
        buffer.labels = state["labels"]
        buffer._random.setstate(state["random_state"])
        return buffer
