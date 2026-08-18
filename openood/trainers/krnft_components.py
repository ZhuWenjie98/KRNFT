"""Reusable building blocks for the KR-NFT training objective.

The original implementation mixed crop selection, negative-label grouping,
loss computation, and progress reporting in one trainer class.  Keeping the
tensor operations here makes the trainer easier to read and test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    """Individual KR-NFT loss terms and their weighted total."""

    positive: torch.Tensor
    negative: torch.Tensor
    knowledge: torch.Tensor
    total: torch.Tensor

    def format(self) -> str:
        """Return a compact message suitable for a training progress bar."""

        return (
            f"loss={self.total.item():.6f} "
            f"positive={self.positive.item():.6f} "
            f"negative={self.negative.item():.6f} "
            f"knowledge={self.knowledge.item():.6f}"
        )


def select_crop_features(
    image_features: torch.Tensor,
    similarities: torch.Tensor,
    num_selected: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the most ID-like and least ID-like crop features.

    Args:
        image_features: Crop features with shape ``[num_crops, feature_dim]``.
        similarities: Similarity of each crop to the target class, shaped
            ``[num_crops]`` or ``[num_crops, 1]``.
        num_selected: Number of positive and negative crops to keep.

    Returns:
        ``(positive_crops, negative_crops, best_crop)``.
    """

    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [num_crops, feature_dim]")

    crop_count = image_features.shape[0]
    if crop_count == 0:
        raise ValueError("at least one crop is required")

    selection_count = min(max(1, num_selected), crop_count)
    flat_similarities = similarities.reshape(-1)
    if flat_similarities.numel() != crop_count:
        raise ValueError("similarities must contain one value per crop")

    ranked_indices = torch.argsort(flat_similarities)
    negative_indices = ranked_indices[:selection_count]
    positive_indices = ranked_indices[-selection_count:]
    best_index = ranked_indices[-1:].contiguous()

    return (
        image_features.index_select(0, positive_indices),
        image_features.index_select(0, negative_indices),
        image_features.index_select(0, best_index),
    )


def build_grouped_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    num_groups: int,
    *,
    shuffle_negative: bool = True,
    apply_softmax: bool = False,
) -> List[torch.Tensor]:
    """Combine positive logits with equally sized groups of negative logits."""

    if positive_logits.ndim != 2 or negative_logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if positive_logits.shape[0] != negative_logits.shape[0]:
        raise ValueError("positive and negative logits must share the batch size")
    if num_groups < 1:
        raise ValueError("num_groups must be positive")

    usable_negative_count = negative_logits.shape[1] - negative_logits.shape[1] % num_groups
    if usable_negative_count == 0:
        raise ValueError("negative logits must contain at least one full group")
    negative_logits = negative_logits[:, :usable_negative_count]

    if shuffle_negative:
        # A local generator avoids changing the caller's global random state.
        generator = torch.Generator(device=negative_logits.device)
        generator.manual_seed(0)
        permutation = torch.randperm(
            usable_negative_count,
            generator=generator,
            device=negative_logits.device,
        )
        negative_logits = negative_logits.index_select(1, permutation)

    group_size = usable_negative_count // num_groups
    grouped_negative = negative_logits.reshape(
        negative_logits.shape[0], num_groups, group_size
    )

    grouped = []
    for group_index in range(num_groups):
        logits = torch.cat(
            [positive_logits, grouped_negative[:, group_index, :]], dim=1
        )
        grouped.append(logits.softmax(dim=1) if apply_softmax else logits)
    return grouped


def compute_krnft_loss(
    prompt_features: torch.Tensor,
    teacher_features: torch.Tensor,
    positive_image_features: torch.Tensor,
    negative_image_features: torch.Tensor,
    positive_targets: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    num_id_classes: int,
    num_groups: int,
    positive_weight: float,
    negative_weight: float,
    knowledge_weight: float,
) -> LossBreakdown:
    """Compute classification, OOD, and knowledge-regularization losses."""

    positive_logits = logit_scale * (positive_image_features @ prompt_features.T)
    negative_logits = logit_scale * (negative_image_features @ prompt_features.T)

    positive_loss = positive_logits.new_zeros(())
    for sample_logits, sample_targets in zip(positive_logits, positive_targets):
        id_logits = sample_logits[:, :num_id_classes]
        ood_logits = sample_logits[:, num_id_classes:]
        grouped = build_grouped_logits(
            id_logits,
            ood_logits,
            num_groups,
            shuffle_negative=True,
            apply_softmax=False,
        )
        repeated_targets = sample_targets.unsqueeze(1).expand(
            -1, len(grouped)
        )
        grouped_logits = torch.stack(grouped, dim=1).reshape(
            -1, grouped[0].shape[-1]
        )
        positive_loss = positive_loss + F.cross_entropy(
            grouped_logits, repeated_targets.reshape(-1)
        )
    positive_loss = positive_loss / positive_logits.shape[0]

    negative_loss = negative_logits.new_zeros(())
    for sample_logits in negative_logits:
        id_logits = sample_logits[:, :num_id_classes]
        ood_logits = sample_logits[:, num_id_classes:]
        grouped = build_grouped_logits(
            id_logits,
            ood_logits,
            num_groups,
            shuffle_negative=True,
            apply_softmax=True,
        )
        id_probabilities = [group[:, :num_id_classes].sum(dim=1) for group in grouped]
        negative_loss = negative_loss + torch.stack(
            [probability.clamp_min(1e-12).log().mean() for probability in id_probabilities]
        ).mean()
    negative_loss = negative_loss / negative_logits.shape[0]

    knowledge_loss = 1.0 - F.cosine_similarity(
        prompt_features, teacher_features, dim=-1
    ).mean()
    total_loss = (
        positive_weight * positive_loss
        + negative_weight * negative_loss
        + knowledge_weight * knowledge_loss
    )

    return LossBreakdown(
        positive=positive_loss,
        negative=negative_loss,
        knowledge=knowledge_loss,
        total=total_loss,
    )
