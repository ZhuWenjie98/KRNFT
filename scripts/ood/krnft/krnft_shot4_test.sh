#!/usr/bin/env bash
set -euo pipefail

# Evaluate a previously trained four-shot KR-NFT checkpoint.
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
output_dir="./results/krnft/imagenet_shot4"

CUDA_VISIBLE_DEVICES="$gpu_id" python main.py \
  --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
  configs/networks/krnft.yml \
  configs/pipelines/train/train_krnft.yml \
  configs/preprocessors/randcrop_preprocessor.yml \
  configs/postprocessors/mcm.yml \
  --dataset.val.batch_size 1 \
  --dataset.test.batch_size 1 \
  --ood_dataset.batch_size 1 \
  --network.name clip_krnft \
  --network.backbone.text_prompt nice \
  --network.backbone.OOD_NUM 10000 \
  --network.backbone.meta_dim 64 \
  --network.pretrained False \
  --trainer.name krnft \
  --postprocessor.name oneoodpromptdevelop \
  --postprocessor.postprocessor_args.tau 1.0 \
  --postprocessor.postprocessor_args.beta 1.0 \
  --postprocessor.postprocessor_args.in_score sum \
  --postprocessor.APS_mode False \
  --optimizer.lr 1.0e-5 \
  --num_gpus 1 \
  --num_workers 6 \
  --merge_option merge \
  --output_dir "$output_dir" \
  --mark shot4
