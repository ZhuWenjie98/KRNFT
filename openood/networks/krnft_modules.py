"""Neural modules specific to the KR-NFT adaptation strategy."""

from __future__ import annotations

import torch
import torch.nn as nn


_LEGACY_PREFIXES = {
    "text_feature_scaling": "id_scale",
    "neg_text_feature_scaling": "ood_scale",
    "text_feature_residuals": "id_shift",
    "neg_text_feature_residuals": "ood_shift",
    "scaling_meta_net": "id_scale_meta",
    "scaling_ex_meta_net": "ood_scale_meta",
    "bias_meta_net": "id_shift_meta",
    "bias_ex_meta_net": "ood_shift_meta",
}


def migrate_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate checkpoints produced by the original monolithic learner."""

    migrated = {}
    for key, value in state_dict.items():
        new_key = key
        for legacy_prefix, current_prefix in _LEGACY_PREFIXES.items():
            if new_key == legacy_prefix or new_key.startswith(f"{legacy_prefix}."):
                new_key = current_prefix + new_key[len(legacy_prefix) :]
                break
        new_key = new_key.replace(".linear1.", ".0.").replace(".linear2.", ".2.")
        migrated[new_key] = value
    return migrated


class DistributionAwareFeatureTransformer(nn.Module):
    """Apply separate image-conditional affine transforms to ID/OOD text.

    The base text features remain frozen.  Learnable scale and shift vectors
    provide the task adaptation, while four small meta-networks add a residual
    conditioned on the current image feature.
    """

    def __init__(self, config, clip_model, base_text_features: torch.Tensor) -> None:
        super().__init__()
        self.id_class_count = int(
            getattr(
                config.backbone,
                "ID_NUM",
                config.num_classes - config.backbone.OOD_NUM,
            )
        )
        self.feature_dim = int(base_text_features.shape[-1])
        visual_dim = int(clip_model.visual.output_dim)
        reduction = max(1, int(config.backbone.meta_dim))
        hidden_dim = max(1, visual_dim // reduction)

        self.register_buffer("base_text_features", base_text_features.detach())
        self.id_scale = nn.Parameter(torch.ones(1, self.feature_dim))
        self.ood_scale = nn.Parameter(torch.ones(1, self.feature_dim))
        self.id_shift = nn.Parameter(torch.zeros(1, self.feature_dim))
        self.ood_shift = nn.Parameter(torch.zeros(1, self.feature_dim))

        self.id_scale_meta = self._make_meta_network(visual_dim, hidden_dim)
        self.ood_scale_meta = self._make_meta_network(visual_dim, hidden_dim)
        self.id_shift_meta = self._make_meta_network(visual_dim, hidden_dim)
        self.ood_shift_meta = self._make_meta_network(visual_dim, hidden_dim)

    def _make_meta_network(self, input_dim: int, hidden_dim: int) -> nn.Sequential:
        network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        return network

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim == 1:
            image_features = image_features.unsqueeze(0)
        if image_features.shape[0] != 1:
            raise ValueError(
                "DistributionAwareFeatureTransformer expects one conditioning image"
            )

        id_scale = self.id_scale + self.id_scale_meta(image_features)
        ood_scale = self.ood_scale + self.ood_scale_meta(image_features)
        id_shift = self.id_shift + self.id_shift_meta(image_features)
        ood_shift = self.ood_shift + self.ood_shift_meta(image_features)

        scale = torch.cat(
            [
                id_scale.expand(self.id_class_count, -1),
                ood_scale.expand(self.base_text_features.shape[0] - self.id_class_count, -1),
            ],
            dim=0,
        )
        shift = torch.cat(
            [
                id_shift.expand(self.id_class_count, -1),
                ood_shift.expand(self.base_text_features.shape[0] - self.id_class_count, -1),
            ],
            dim=0,
        )
        return scale * self.base_text_features + shift
