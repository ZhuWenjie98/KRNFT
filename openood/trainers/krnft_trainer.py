"""Trainer for Knowledge Regularized Negative Feature Tuning (KR-NFT)."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.networks.clip import clip
from openood.networks.clip.clip import tokenize
from openood.networks.clip_fixed_ood_prompt import imagenet_classes
from openood.utils import Config

from .krnft_components import LossBreakdown, compute_krnft_loss, select_crop_features


def load_clip_to_cpu(backbone_name: str) -> nn.Module:
    """Load an OpenAI CLIP checkpoint without allocating it on the GPU."""

    model_path = clip._download(clip._MODELS[backbone_name])
    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict)


class AverageMeter:
    """Track a running mean for scalar training metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += value * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)


class KRNFTTrainer:
    """Optimize the lightweight text-feature transformations used by KR-NFT.

    The CLIP image and text encoders are frozen. Only the task-residual learner
    attached to ``net.model`` receives gradients, which keeps training
    efficient and preserves the pre-trained representation.
    """

    def __init__(self, net: nn.Module, train_loader: DataLoader, config: Config) -> None:
        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        learner = self.net.model.taskres_prompt_learner
        self.optimizer = torch.optim.AdamW(
            learner.parameters(),
            lr=float(config.optimizer.lr),
            weight_decay=float(getattr(config.optimizer, "weight_decay", 0.0)),
        )
        self.selection_count = int(config.trainer.trainer_args.n_selection)
        self.num_id_classes = int(config.n_cls)
        self.num_groups = int(config.n_group)
        self.positive_weight = float(config.lam_in)
        self.negative_weight = float(config.lam_out)
        self.knowledge_weight = float(config.lam_kd)

        self.net.to(self.device, dtype=torch.float32)
        self.clip_model = getattr(self.net, "clip_model", None)
        if self.clip_model is None:
            self.clip_model = load_clip_to_cpu(config.backbone.name)
        self.clip_model = self.clip_model.to(self.device, dtype=torch.float32)
        self.clip_model.eval()
        self.label_features = self._encode_label_features()

    def setup(self) -> None:
        """Keep the trainer API compatible with other OpenOOD trainers."""

    @torch.no_grad()
    def _encode_label_features(self) -> torch.Tensor:
        """Encode one vanilla prompt per ImageNet class for crop ranking."""

        prompts = [f"a photo of a {label}." for label in imagenet_classes]
        text_features = self.clip_model.encode_text(tokenize(prompts).to(self.device))
        text_features = text_features.float()
        return text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def train_epoch(self, epoch_idx: int):
        """Run one complete epoch and persist the learner checkpoint."""

        self.net.train()
        loss_meter = AverageMeter()
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx:03d}",
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        )

        for batch in progress:
            images = batch["data"].to(self.device)
            targets = batch["label"].to(self.device)
            positive_features, negative_features, positive_targets, conditioning = (
                self._build_training_features(images, targets)
            )

            prompt_features = self.net.model.taskres_prompt_learner(conditioning)
            prompt_features = torch.nn.functional.normalize(prompt_features, dim=-1)
            teacher_features = torch.nn.functional.normalize(self.net.text_features, dim=-1)
            losses = compute_krnft_loss(
                prompt_features,
                teacher_features,
                positive_features,
                negative_features,
                positive_targets,
                self.net.logit_scale.exp(),
                num_id_classes=self.num_id_classes,
                num_groups=self.num_groups,
                positive_weight=self.positive_weight,
                negative_weight=self.negative_weight,
                knowledge_weight=self.knowledge_weight,
            )

            self.optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            self.optimizer.step()

            loss_meter.update(losses.total.detach().item())
            progress.set_postfix_str(losses.format())

        self._save_checkpoint()
        metrics = {"epoch_idx": epoch_idx, "loss": self._reduce_metric(loss_meter.average)}
        return self.net, metrics

    @torch.no_grad()
    def _build_training_features(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create positive/negative crop features and their ID labels."""

        positive_batches = []
        negative_batches = []
        target_batches = []
        conditioning_batches = []

        for image_crops, target in zip(images, targets):
            crop_features = self.net.get_image_features(image_crops)
            crop_features = torch.nn.functional.normalize(crop_features, dim=-1)
            class_feature = self.label_features[target.long()]
            similarities = crop_features @ class_feature
            positive, negative, best = select_crop_features(
                crop_features, similarities, self.selection_count
            )

            positive_batches.append(positive)
            negative_batches.append(negative)
            target_batches.append(
                torch.full(
                    (positive.shape[0],),
                    int(target),
                    dtype=torch.long,
                    device=self.device,
                )
            )
            conditioning_batches.append(best.squeeze(0))

        return (
            torch.stack(positive_batches),
            torch.stack(negative_batches),
            torch.stack(target_batches),
            torch.stack(conditioning_batches),
        )

    def _save_checkpoint(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.output_dir, "model_checkpoint.pth")
        torch.save(
            {
                "taskres_prompt_learner_state_dict": self.net.model.taskres_prompt_learner.state_dict()
            },
            checkpoint_path,
        )

    @staticmethod
    def _reduce_metric(value: float) -> float:
        """Average a scalar across distributed workers."""

        reduced = comm.gather(value)
        return float(np.mean(reduced))

    # Keep these names for downstream scripts that imported the old trainer API.
    def get_in_out(self, clip_model, model, labels, images, targets):
        del clip_model, model, labels
        positive, negative, positive_targets, conditioning = self._build_training_features(
            images, targets
        )
        return positive, negative, positive_targets, [], conditioning

    def get_loss(
        self,
        prompt_features,
        teacher_features,
        positive_features,
        negative_features,
        positive_targets,
        negative_targets,
        logit_scale,
    ) -> tuple[torch.Tensor, str]:
        del negative_targets
        losses: LossBreakdown = compute_krnft_loss(
            prompt_features,
            teacher_features,
            positive_features,
            negative_features,
            positive_targets,
            logit_scale,
            num_id_classes=self.num_id_classes,
            num_groups=self.num_groups,
            positive_weight=self.positive_weight,
            negative_weight=self.negative_weight,
            knowledge_weight=self.knowledge_weight,
        )
        return losses.total, losses.format()
