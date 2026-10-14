"""Exponential-moving-average vector quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    active_fraction: torch.Tensor


class EMAVectorQuantizer(nn.Module):
    """Nearest-neighbour quantizer whose codebook is updated with EMA."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iterations: int = 10,
        dead_code_threshold: float = 0.5,
        update_sample_size: int | None = None,
    ) -> None:
        super().__init__()
        if codebook_size < 2 or embedding_dim < 1:
            raise ValueError("codebook_size and embedding_dim are invalid")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold cannot be negative")
        if update_sample_size is not None and update_sample_size <= 0:
            raise ValueError("update_sample_size must be positive or None")

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        self.kmeans_iterations = kmeans_iterations
        self.dead_code_threshold = dead_code_threshold
        self.update_sample_size = update_sample_size

        codebook = torch.randn(codebook_size, embedding_dim)
        codebook /= embedding_dim**0.5
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", codebook.clone())
        self.register_buffer("initialized", torch.tensor(not kmeans_init))

    @staticmethod
    def _distances(inputs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return (
            inputs.square().sum(dim=1, keepdim=True)
            - 2.0 * inputs @ codebook.T
            + codebook.square().sum(dim=1).unsqueeze(0)
        )

    @torch.no_grad()
    def _sample_inputs(self, flat_inputs: torch.Tensor, count: int) -> torch.Tensor:
        samples = flat_inputs.shape[0]
        if samples >= count:
            return flat_inputs[torch.randperm(samples, device=flat_inputs.device)[:count]]
        repeats = (count + samples - 1) // samples
        repeated = flat_inputs.repeat(repeats, 1)[:count]
        noise = torch.randn_like(repeated) * 1e-4
        return repeated + noise

    @torch.no_grad()
    def _initialize_codebook(self, flat_inputs: torch.Tensor) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if rank == 0:
            centroids = self._sample_inputs(flat_inputs, self.codebook_size)
            for _ in range(self.kmeans_iterations):
                assignments = self._distances(flat_inputs, centroids).argmin(dim=1)
                counts = torch.bincount(
                    assignments, minlength=self.codebook_size
                ).to(flat_inputs.dtype)
                sums = torch.zeros_like(centroids)
                sums.index_add_(0, assignments, flat_inputs)
                populated = counts > 0
                centroids[populated] = sums[populated] / counts[populated, None]
                empty = ~populated
                if empty.any():
                    centroids[empty] = self._sample_inputs(
                        flat_inputs, int(empty.sum())
                    )

            assignments = self._distances(flat_inputs, centroids).argmin(dim=1)
            counts = torch.bincount(
                assignments, minlength=self.codebook_size
            ).to(flat_inputs.dtype)
            self.codebook.copy_(centroids)
            self.ema_cluster_size.copy_(counts.clamp_min(1.0))
            self.ema_embed_sum.copy_(
                centroids * self.ema_cluster_size.unsqueeze(1)
            )

        if distributed:
            dist.broadcast(self.codebook, src=0)
            dist.broadcast(self.ema_cluster_size, src=0)
            dist.broadcast(self.ema_embed_sum, src=0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def initialize_codebook(self, inputs: torch.Tensor) -> None:
        """Fit code vectors to a representative continuous-latent sample."""
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"last input dimension must be {self.embedding_dim}, "
                f"received {inputs.shape[-1]}"
            )
        flat_inputs = inputs.reshape(-1, self.embedding_dim).float()
        if not len(flat_inputs):
            raise ValueError("cannot initialize a codebook from no inputs")
        self._initialize_codebook(flat_inputs)

    @torch.no_grad()
    def _update_codebook(
        self, flat_inputs: torch.Tensor, flat_indices: torch.Tensor
    ) -> None:
        if (
            self.update_sample_size is not None
            and len(flat_inputs) > self.update_sample_size
        ):
            selection = torch.randperm(
                len(flat_inputs), device=flat_inputs.device
            )[: self.update_sample_size]
            flat_inputs = flat_inputs[selection]
            flat_indices = flat_indices[selection]
        counts = torch.bincount(
            flat_indices, minlength=self.codebook_size
        ).to(flat_inputs.dtype)
        embed_sum = torch.zeros_like(self.ema_embed_sum)
        embed_sum.index_add_(0, flat_indices, flat_inputs)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(embed_sum)

        self.ema_cluster_size.mul_(self.decay).add_(
            counts, alpha=1.0 - self.decay
        )
        self.ema_embed_sum.mul_(self.decay).add_(
            embed_sum, alpha=1.0 - self.decay
        )

        total = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + self.epsilon)
            / (total + self.codebook_size * self.epsilon)
            * total
        )
        self.codebook.copy_(self.ema_embed_sum / smoothed_size.unsqueeze(1))

        dead = self.ema_cluster_size < self.dead_code_threshold
        if dead.any():
            replacements = self._sample_inputs(flat_inputs, int(dead.sum()))
            self.codebook[dead] = replacements
            self.ema_cluster_size[dead] = 1.0
            self.ema_embed_sum[dead] = replacements

    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"last input dimension must be {self.embedding_dim}, "
                f"received {inputs.shape[-1]}"
            )

        input_dtype = inputs.dtype
        flat_inputs = inputs.reshape(-1, self.embedding_dim).float()
        if self.training and not self.initialized:
            self._initialize_codebook(flat_inputs.detach())
        codebook = self.codebook.float()
        distances = self._distances(flat_inputs, codebook)
        flat_indices = distances.argmin(dim=1)
        nearest = F.embedding(flat_indices, self.codebook).to(input_dtype)

        if self.training:
            self._update_codebook(flat_inputs.detach(), flat_indices)

        nearest = nearest.reshape_as(inputs)
        quantized = inputs + (nearest - inputs).detach()
        commitment_loss = self.commitment_weight * F.mse_loss(
            inputs.float(), nearest.detach().float()
        )

        counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()

        return QuantizerOutput(
            quantized=quantized,
            indices=flat_indices.reshape(inputs.shape[:-1]),
            commitment_loss=commitment_loss,
            perplexity=entropy.exp(),
            active_fraction=nonzero.float().mean(),
        )
