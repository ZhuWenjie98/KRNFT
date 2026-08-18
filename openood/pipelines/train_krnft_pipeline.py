"""End-to-end training and evaluation pipeline for KR-NFT."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.networks.krnft_modules import migrate_legacy_state_dict
from openood.postprocessors import get_postprocessor
from openood.trainers import get_trainer
from openood.utils import setup_logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task_residual_checkpoint(config, network):
    """Restore only the trainable KR-NFT text-feature learner."""

    checkpoint_path = os.path.join(config.output_dir, "model_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    learner_state = migrate_legacy_state_dict(
        checkpoint["taskres_prompt_learner_state_dict"]
    )
    network.model.taskres_prompt_learner.load_state_dict(learner_state, strict=False)
    return network


class TrainKRNFTPipeline:
    """Coordinate data loading, training, checkpointing, and OOD evaluation."""

    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> None:
        setup_logger(self.config)
        set_seed(int(self.config.seed))

        loaders = get_dataloader(self.config)
        ood_loaders = get_ood_dataloader(self.config)
        network = get_network(self.config.network)

        evaluator = get_evaluator(self.config)
        postprocessor = get_postprocessor(self.config)
        postprocessor.setup(network, loaders, ood_loaders)

        checkpoint_path = os.path.join(self.config.output_dir, "model_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            network = load_task_residual_checkpoint(self.config, network)
            print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
        else:
            trainer = get_trainer(
                network,
                loaders["train"],
                loaders.get("val"),
                self.config,
            )
            trainer.setup()
            print("Start training...", flush=True)
            for epoch_index in range(1, int(self.config.optimizer.num_epochs) + 1):
                network, metrics = trainer.train_epoch(epoch_index)
                print(
                    f"Epoch {metrics['epoch_idx']} complete; "
                    f"mean loss={metrics['loss']:.6f}",
                    flush=True,
                )

        print("Start OOD evaluation...", flush=True)
        evaluator.eval_ood(network, loaders, ood_loaders, postprocessor)
        print("Completed.", flush=True)
