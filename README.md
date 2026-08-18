# KR-NFT

### Knowledge Regularized Negative Feature Tuning of Vision-Language Models for Out-of-Distribution Detection

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2507.19847-b31b1b.svg)](https://arxiv.org/abs/2507.19847)
[![Conference](https://img.shields.io/badge/ACM%20MM-2025-00599c.svg)](https://doi.org/10.1145/3746027.3755120)

Official implementation of **Knowledge Regularized Negative Feature Tuning (KR-NFT)**, an efficient adaptation method for CLIP-style vision-language models. KR-NFT improves OOD detection on the training classes while preserving the pre-trained knowledge required to generalize to unseen classes and image styles.

> Paper: Wenjie Zhu, Yabin Zhang, Xin Jin, Wenjun Zeng, and Lei Zhang. *Knowledge Regularized Negative Feature Tuning of Vision-Language Models for Out-of-Distribution Detection*. ACM Multimedia 2025.

## Why KR-NFT?

Prompt-tuned VLMs can become over-specialized to the few-shot training classes. KR-NFT addresses this failure mode with three complementary ideas:

1. **Negative Feature Tuning (NFT)** directly applies learnable element-wise scale and shift factors to frozen text features, avoiding back-propagation through the text encoder.
2. **Image-conditional modulation** uses a lightweight meta-network to adapt the transformation to each input image, reducing sensitivity to class and style shifts.
3. **Knowledge regularization (KR)** maximizes the cosine similarity between the original and tuned text features, balancing task adaptation against pre-trained knowledge forgetting.

The training objective is:

```text
L = L_positive + lambda_1 * L_negative + lambda_2 * L_knowledge
```

Positive crops are trained with cross-entropy, negative crops minimize the ID probability mass, and the knowledge term constrains the tuned text features to remain close to the original CLIP features.


## Reported results

The paper evaluates a ViT-B/16 model trained with four shots per ImageNet-1K class and 10,000 negative labels.

### Unseen classes

| Method | CIFAR-10 AUROC | CIFAR-10 FPR95 | CIFAR-100 AUROC | CIFAR-100 FPR95 | Fine-grained AUROC | Fine-grained FPR95 | Average FPR95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LAPT | 95.21 | 17.26 | 78.94 | 57.11 | 89.23 | 46.91 | 40.43 |
| **KR-NFT** | **96.82** | **11.41** | **80.95** | **53.24** | **89.99** | **40.33** | **34.99** |

### Unseen styles

| Method | ImageNet-S FPR95 | ImageNet-A FPR95 | ImageNet-R FPR95 | ImageNet-V2 FPR95 | Average FPR95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAPT | 51.87 | 51.47 | 47.42 | 28.84 | 44.90 |
| **KR-NFT** | **24.81** | **38.46** | **20.12** | **29.13** | **28.13** |

The paper reports an average FPR95 improvement of **5.44 percentage points on unseen classes** and **16.77 percentage points on unseen styles** over the closest tuning baseline.

## Repository layout

```text
.
├── configs/
│   ├── datasets/                  # OpenOOD dataset definitions
│   ├── networks/krnft.yml         # CLIP + KR-NFT architecture
│   └── pipelines/train_krnft.yml  # Optimizer and loss weights
├── openood/
│   ├── networks/clip_krnft.py     # CLIP backbone and text-feature learner
│   ├── networks/krnft_modules.py  # Distribution-aware feature transform
│   ├── trainers/krnft_components.py # Crop selection and KR-NFT losses
│   ├── trainers/krnft_trainer.py  # Training loop and checkpointing
│   └── pipelines/train_krnft_pipeline.py
├── scripts/ood/krnft/             # Reproducible four-shot commands
├── 2507.19847v2.pdf               # Paper
└── README.md
```

## Installation

This repository is an OpenOOD extension. Install the upstream runtime first:

```bash
pip install git+https://github.com/YBZH/OpenOOD-VLM
```

The training scripts expect to be launched from an OpenOOD-VLM workspace that provides `main.py`, or from a workspace where this repository has been copied into the corresponding OpenOOD package. A CUDA-capable PyTorch installation is required for the full CLIP experiment.

## Data and checkpoints

The default OpenOOD paths are:

```text
data/
├── benchmark_imglist/
├── images_classic/
└── images_largescale/
results/
└── checkpoints/
```

For the paper protocol, prepare:

- **ID:** ImageNet-1K, four shots per class;
- **Training OOD proxies:** 10,000 negative text labels;
- **ImageNet OOD:** iNaturalist, SUN, Places, and Textures;
- **Unseen classes:** CIFAR-10, CIFAR-100, and four fine-grained datasets;
- **Unseen styles:** ImageNet-Sketch, ImageNet-A, ImageNet-R, and ImageNet-V2.

Follow the [OpenOOD download instructions](https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download) for dataset preparation. ImageNet-1K training images must be obtained from the official source.

## Quick start

### Train

The default configuration follows the paper: 256 random crops per image, 32 selected positive/negative crops, three epochs, AdamW with learning rate `1e-5`, `lambda_1=0.3`, and `lambda_2=100`.

```bash
bash scripts/ood/krnft/krnft_shot4_train.sh
```

Set `CUDA_VISIBLE_DEVICES` to select a GPU and edit `output_dir` in the script if you want to use a different experiment directory.

### Evaluate

```bash
bash scripts/ood/krnft/krnft_shot4_test.sh
```

The evaluator writes the standard OpenOOD OOD metrics, including AUROC and FPR95, to the configured result directory.

### Run with custom settings

All experiment parameters are exposed through the YAML files and command-line overrides:

```bash
python main.py \
  --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
  configs/networks/krnft.yml \
  configs/pipelines/train/train_krnft.yml \
  configs/preprocessors/randcrop_preprocessor.yml \
  configs/postprocessors/mcm.yml \
  --dataset.train.few_shot 4 \
  --network.backbone.OOD_NUM 10000 \
  --optimizer.lr 1.0e-5
```

## Implementation notes

- `openood/networks/krnft_modules.py` implements the image-conditional, distribution-aware affine transformation from the paper.
- `openood/trainers/krnft_components.py` contains the numerical building blocks: crop ranking, negative-label grouping, and the three KR-NFT loss terms.
- `openood/trainers/krnft_trainer.py` owns data movement, optimization, metric reduction, and checkpoint persistence. Only the task-residual learner is optimized.
- `openood/pipelines/train_krnft_pipeline.py` handles reproducible seeding, data/network construction, checkpoint reuse, and OOD evaluation.
- The network keeps the CLIP image encoder and original text features frozen. The distribution-aware learner maintains separate transformations for positive ID labels and negative labels.

## Citation

```bibtex
@inproceedings{zhu2025krnft,
  title     = {Knowledge Regularized Negative Feature Tuning of Vision-Language Models for Out-of-Distribution Detection},
  author    = {Zhu, Wenjie and Zhang, Yabin and Jin, Xin and Zeng, Wenjun and Zhang, Lei},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  year      = {2025},
  doi       = {10.1145/3746027.3755120},
  eprint    = {2507.19847},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

This implementation builds on [OpenOOD](https://github.com/Jingkang50/OpenOOD) and CLIP. Please follow the licenses of the upstream projects and the dataset providers.
