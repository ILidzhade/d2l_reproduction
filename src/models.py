"""Define the conditional VAEs, latent classifier, and VAE losses."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import Config


class ConditionalVAE(nn.Module):
    """Label-free posterior and classifier; observed-label-conditioned decoder."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.image_size = cfg.image_size
        self.spatial = cfg.image_size // 16
        layers: list[nn.Module] = []
        incoming = 3
        for outgoing in (32, 64, 96, 128):
            layers.extend([
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.ReLU(), nn.MaxPool2d(2),
            ])
            incoming = outgoing
        self.encoder = nn.Sequential(*layers)
        flattened = 128 * self.spatial**2
        self.mu = nn.Linear(flattened, cfg.latent_dim)
        self.logvar = nn.Linear(flattened, cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, cfg.total_classes)
        self.condition = nn.Embedding(cfg.total_classes, cfg.condition_dim)
        self.decoder_input = nn.Linear(
            cfg.latent_dim + cfg.condition_dim, 64 * self.spatial**2
        )
        decoder: list[nn.Module] = []
        for incoming, outgoing in ((64, 48), (48, 32), (32, 16), (16, 3)):
            decoder.append(nn.ConvTranspose2d(incoming, outgoing, 4, 2, 1))
            if outgoing != 3:
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def posterior(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return the diagonal-Gaussian posterior mean and clamped log variance."""
        features = self.encoder(images).flatten(1)
        return self.mu(features), self.logvar(features).clamp(-12, 8)

    def encode(self, images: Tensor) -> Tensor:
        """Encode images as deterministic posterior-mean latent features."""
        return self.posterior(images)[0]

    def classify_features(self, features: Tensor) -> Tensor:
        """Return classifier logits for latent feature vectors."""
        return self.head(features)

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Decode latent vectors conditioned on observed real-class labels."""
        features = torch.cat((z, self.condition(labels)), dim=-1)
        seed = self.decoder_input(features).reshape(-1, 64, self.spatial, self.spatial)
        images = self.decoder(seed)
        return F.interpolate(
            images, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).sigmoid()

    def reconstruct(self, images: Tensor, labels: Tensor) -> Tensor:
        """Reconstruct images from posterior means and their observed labels."""
        return self.decode(self.encode(images), labels)

    def forward(self, images: Tensor) -> Tensor:
        """Return class-slot logits for input images."""
        return self.classify_features(self.encode(images))


def _group_count(channels: int) -> int:
    """Choose the largest supported GroupNorm divisor for a channel count."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualDownsample(nn.Module):
    """Downsample spatial features with a normalized residual convolution block."""
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(outgoing), outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(_group_count(outgoing), outgoing),
        )
        self.skip = nn.Conv2d(incoming, outgoing, 1, stride=2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return features downsampled by a factor of two."""
        return F.silu(self.main(inputs) + self.skip(inputs))


class ResidualUpsample(nn.Module):
    """Resize spatial features and refine them with a residual convolution block."""
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(_group_count(outgoing), outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(_group_count(outgoing), outgoing),
        )
        self.skip = nn.Conv2d(incoming, outgoing, 1)

    def forward(self, inputs: Tensor, size: tuple[int, int]) -> Tensor:
        """Return residual features resized to the requested spatial dimensions."""
        resized = F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)
        return F.silu(self.main(resized) + self.skip(resized))


class BalancedConditionalVAE(ConditionalVAE):
    """Matched residual encoder/decoder with an exact-size image path."""

    def __init__(self, cfg: Config) -> None:
        nn.Module.__init__(self)
        self.image_size = cfg.image_size
        self.spatial_sizes = self._spatial_sizes(cfg.image_size)
        widths = (64, 128, 192, 256)
        blocks: list[nn.Module] = []
        incoming = 3
        for outgoing in widths:
            blocks.append(ResidualDownsample(incoming, outgoing))
            incoming = outgoing
        self.encoder = nn.Sequential(*blocks)
        seed_size = self.spatial_sizes[-1]
        flattened = widths[-1] * seed_size**2
        self.mu = nn.Linear(flattened, cfg.latent_dim)
        self.logvar = nn.Linear(flattened, cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, cfg.total_classes)
        self.condition = nn.Embedding(cfg.total_classes, cfg.condition_dim)
        self.decoder_input = nn.Linear(
            cfg.latent_dim + cfg.condition_dim, widths[-1] * seed_size**2
        )
        self.decoder = nn.ModuleList([
            ResidualUpsample(256, 192),
            ResidualUpsample(192, 128),
            ResidualUpsample(128, 64),
            ResidualUpsample(64, 32),
        ])
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    @staticmethod
    def _spatial_sizes(image_size: int) -> tuple[int, int, int, int, int]:
        """Return exact spatial sizes produced by four stride-two encoder stages."""
        sizes = [image_size]
        for _ in range(4):
            sizes.append((sizes[-1] + 1) // 2)
        return sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Decode conditioned latents through the exact-size residual image path."""
        features = torch.cat((z, self.condition(labels)), dim=-1)
        seed_size = self.spatial_sizes[-1]
        decoded = self.decoder_input(features).reshape(-1, 256, seed_size, seed_size)
        for block, size in zip(
            self.decoder, reversed(self.spatial_sizes[:-1]), strict=True
        ):
            decoded = block(decoded, (size, size))
        return self.output(decoded).sigmoid()


def build_model(cfg: Config) -> ConditionalVAE:
    """Construct the configured legacy or balanced conditional VAE.

    Args:
        cfg: Configuration selecting architecture dimensions and class capacity.

    Returns:
        An uninitialized conditional VAE matching the configuration.
    """
    if cfg.model_variant == "balanced":
        return BalancedConditionalVAE(cfg)
    return ConditionalVAE(cfg)


def vae_losses(
    model: ConditionalVAE, images: Tensor, labels: Tensor,
    mu: Tensor, logvar: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute mean-pixel L1 reconstruction and mean-dimension KL losses.

    Args:
        model: Conditional VAE used to decode a posterior sample.
        images: Target image batch in the range zero to one.
        labels: Observed labels used to condition the decoder.
        mu: Posterior mean for each input image.
        logvar: Posterior diagonal log variance for each input image.

    Returns:
        Scalar reconstruction and KL-divergence losses.
    """
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    reconstruction = F.l1_loss(model.decode(z, labels), images)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return reconstruction, kl
