# Leveraging Hierarchies via multimodal prompt learning for Fine-Grained Image Recognition

This repository contains the implementation of a hierarchical prompt learning framework for fine-grained image classification using the CLIP vision-language model. The project investigates the use of hierarchical information (coarse and fine labels) to enhance classification accuracy in both few-shot (16-shot) and fully supervised (all-shot) settings.

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

## Project Overview

Fine-grained image classification involves distinguishing visually similar subcategories within a broader category (e.g., different species of flowers or animals). This task is challenging in few-shot settings due to limited training data, which hinders learning discriminative features. The CLIP model [1], pretrained on image-text pairs, offers strong zero-shot performance but struggles with fine-grained tasks due to subtle class distinctions.

This project addresses this challenge by leveraging hierarchical information, where fine-grained classes are grouped into coarse-grained superclasses (e.g., "bear" under "large carnivore"). We implement eight methods to incorporate hierarchical context into CLIP-based prompt learning, inspired by CoOp [2] and CoCoOp [3].

### Key Features

- Support for both 16-shot and all-shot learning settings
- Eight different methods for hierarchical prompt learning (more coming up)
- Distributed training support
- Automatic mixed precision (AMP)
- Early stopping
- Curriculum learning
- Weights & Biases (WandB) integration for logging and visualization

## Methodology

The project implements eight methods to integrate hierarchical information into CLIP-based fine-grained classification. All methods use CLIP (ViT-B/32) as the base model, with learnable prompts optimized via a meta-network conditioned on image features.

### Implemented Methods

1. **plain_fine**
   - Zero-shot CLIP with fixed fine-grained prompts (A photo of a "class")
   - Trained with fine-grained cross-entropy loss
   - Baseline for zero-shot performance

2. **plain_fine_coarse**
   - Zero-shot CLIP with fixed prompts (A photo of a "class" of type "coarse_class") incorporating coarse context
   - Trained with loss_fine only
   - Tests impact of static coarse context

3. **fine_only**
   - Learnable fine-grained prompts
   - No coarse information used
   - Baseline for prompt learning without hierarchy

4. **fine_coarse_concat**
   - Concatenates coarse context into learnable fine-grained prompts
   - Trained with loss_fine and loss_coarse
   - Returns fine-grained logits during inference

5. **fine_coarse_separate**
   - Separate learnable fine and coarse prompts
   - Trained with loss_fine and loss_coarse
   - Emphasizes explicit coarse supervision

6. **fine_coarse_concat_cross**
   - Extends fine_coarse_concat with cross-attention
   - Captures hierarchical dependencies
   - Trained with loss_fine and loss_coarse

7. **fine_context**
   - Combines separate fine and coarse prompts with cross-attention
   - Explicitly models fine-coarse interactions
   - Trained with loss_fine and loss_coarse

8. **fine_coarse_hierarchical**
   - Novel approach integrating four techniques:
     - Hierarchical Image Feature Enhancement
     - Hierarchical Prompt Regularization
     - Hierarchical Contrastive Loss
     - Coarse-to-Fine Curriculum Learning
   - Trained with combined loss: `total_loss = loss_fine + α·loss_coarse + β·prompt_reg_loss + γ·contrastive_loss`
   - Where α = 0.5, β = 0.1, γ = 0.1

## Datasets

The framework is evaluated on three datasets with hierarchical labels:

| Dataset | Fine Classes | Coarse Classes | Total Images | 16-Shot Images | All-Shot Images |
|---------|-------------|----------------|--------------|----------------|-----------------|
| CIFAR-100 | 100 | 20 | 60,000 | 1,600 | 50,000 |
| EuroSAT | 10 | 5 | 27,000 | 160 | ~18,900 |
| Flowers102 | 102 | ~40 | 8,189 | 1,632 | 5,989 |

## Installation

### Prerequisites

- Python 3.11
- PyTorch 2.0 or later
- CUDA-enabled GPU (optional, recommended for faster training)
- WandB account for logging (optional, but required for result tracking)

### Steps

1. Clone the Repository:
   ```bash
   git clone https://github.com/georgian4049/hierarchical-prompt-learning.git
   cd hierarchical-prompt-learning
   ```

2. Create a Virtual Environment:
   ```bash
   python -m venv clip_env
   source clip_env/bin/activate  # On Windows: clip_env\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set Up WandB:
   ```bash
   pip install wandb
   wandb login
   ```

5. Download Datasets:
   - CIFAR-100: Automatically downloaded via PyTorch's torchvision.datasets
   - EuroSAT: Automatically downloaded via PyTorch's torchvision.datasets
   - Flowers102: Automatically downloaded via PyTorch's torchvision.datasets

6. Set Up Storage:
   ```bash
   # Create directories for checkpoints and logs
   mkdir -p <your-preferred-path>/checkpoints
   mkdir -p <your-preferred-path>/wandb_logs
   
   # Configure WandB paths (optional but recommended for large artifacts)
   # Note: WandB artifacts can be large, so it's recommended to store them in a location with sufficient storage
   export WANDB_DIR=<your-preferred-path>/wandb_logs
   export WANDB_CACHE_DIR=<your-preferred-path>/wandb_cache
   export WANDB_CONFIG_DIR=<your-preferred-path>/wandb_config
   ```

   > **Note on Storage Configuration**: 
   > - Choose storage locations with sufficient disk space, especially for WandB artifacts which can be large
   > - For optimal performance, consider using high-speed storage (e.g., SSD) for checkpoints
   > - The paths can be customized based on your system configuration and storage preferences
   > - Ensure the chosen directories have appropriate read/write permissions

## Usage

### Running an Experiment

To train a model, use the main.py script with command-line arguments or a configuration file:

```bash
python main.py --dataset CIFAR --model_type fine_coarse_hierarchical --shots 16
```

### Command-Line Arguments

- `--dataset`: Dataset name (CIFAR, EuroSAT, Flowers102)
- `--model_type`: Model type (plain_fine, plain_fine_coarse, fine_only, etc.)
- `--shots`: Number of shots (16 for 16-shot, all for all-shot)
- `--config`: Path to configuration file (default: config.yaml)

### Configuration File

Example `config.yaml`:

```yaml
MODEL:
  TYPE: fine_coarse_hierarchical
  PRECISION: amp
  COARSE_LOSS_WEIGHT: 0.5
  PROMPT_REG_WEIGHT: 0.1
  CONTRASTIVE_WEIGHT: 0.1
  CONTRASTIVE_MARGIN: 1.0
DATASET:
  NAME: CIFAR
  BATCH_SIZE: 16
  SHOTS: 16
  NUM_WORKERS: 4
TRAINER:
  LR: 0.001
  EPOCHS: 100
  EARLY_STOPPING: True
  PATIENCE: 10
  MIN_DELTA: 0.001
  MOVING_AVG_WINDOW: 5
  CURRICULUM_EPOCHS: 10
  DISTRIBUTED: False
ARTIFACT:
  NAME: vit-32-CIFAR-fine_coarse_hierarchical-16shots-bs-16
```

## Directory Structure

```
coarse_prompt_learning/
├── configs/               # Configuration files for different experiments
├── datasets/             # Dataset handling and processing
├── models/
│   ├── custom_clip.py    # Custom CLIP implementation with hierarchical features
│   ├── prompt_learner.py # Prompt learning implementation
│   └── zeroshot.py      # Zero-shot baseline implementation
├── trainers/
│   └── cocoop_trainer.py # Trainer implementation for prompt learning
├── utils/
│   └── data_utils.py    # Utilities for data processing and handling
├── main.py              # Entry point for training and evaluation
├── requirements.txt     # Python dependencies
├── settings.py         # Project settings and configurations
└── README.md           # Project documentation
```

## Results

### 16-Shot Setting

| Method | CIFAR-100 | EuroSAT | Flowers102 |
|--------|-----------|---------|------------|
| plain_fine | ...loading | ...loading | ...loading |
| plain_fine_coarse | ...loading | ...loading | ...loading |
| fine_only | ...loading | ...loading | ...loading |
| fine_coarse_concat | ...loading | ...loading | ...loading |
| fine_coarse_separate | ...loading | ...loading | ...loading |
| fine_coarse_concat_cross | ...loading | ...loading | ...loading |
| fine_context | ...loading | ...loading | ...loading |
| fine_coarse_hierarchical | ...loading | ...loading | ...loading |

### All-Shot Setting

| Method | CIFAR-100 | EuroSAT | Flowers102 |
|--------|-----------|---------|------------|
| plain_fine | ...loading | ...loading | ...loading |
| plain_fine_coarse | ...loading | ...loading | ...loading |
| fine_only | ...loading | ...loading | ...loading |
| fine_coarse_concat | ...loading | ...loading | ...loading |
| fine_coarse_separate | ...loading | ...loading | ...loading |
| fine_coarse_concat_cross | ...loading | ...loading | ...loading |
| fine_context | ...loading | ...loading | ...loading |
| fine_coarse_hierarchical | ...loading | ...loading | ...loading |

> **Note**: Results will be updated as experiments are completed. Check back later for the actual performance metrics.

## Troubleshooting

### Common Issues

1. **RuntimeError: Missing key(s) in state_dict**
   - **Cause**: Attempting to load a checkpoint from a different model type
   - **Fix**:
     ```bash
     rm -rf ~/../../work/ML/shekhar/checkpoints
     ```
     - Ensure ARTIFACT.NAME in config.yaml is unique
     - Verify MODEL.TYPE matches the checkpoint's model type

2. **Low Fine-Grained Accuracy**
   - **Cause**: Suboptimal hyperparameters or insufficient training
   - **Fix**:
     - Tune COARSE_LOSS_WEIGHT (0.1–1.0)
     - Adjust PROMPT_REG_WEIGHT (0.01–0.5)
     - Modify CONTRASTIVE_WEIGHT (0.01–0.5)
     - Increase CURRICULUM_EPOCHS (10–15)

3. **WandB Artifact Conflicts**
   - **Cause**: Reusing ARTIFACT.NAME across experiments
   - **Fix**:
     ```bash
     wandb artifact delete vit-32-<dataset>-<model_type>-<shots>-shots-bs-<batch_size>:latest
     ```
     - Use a unique ARTIFACT.NAME (e.g., include timestamp)

## References

1. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Clark, J. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. [Link](https://arxiv.org/abs/2103.00020)

2. Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Learning to Prompt for Vision-Language Models. International Journal of Computer Vision, 130(9), 2337–2348. [Link](https://arxiv.org/abs/2109.01134)

3. Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Conditional Prompt Learning for Vision-Language Models. arXiv preprint arXiv:2203.05557. [Link](https://arxiv.org/abs/2203.05557)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact mailtoayushshekhar@gmail.com.

### Monitoring Training

Training progress is logged to WandB, including:
- train_loss_fine, train_loss_coarse, train_prompt_reg_loss, train_contrastive_loss
- eval_loss_fine, eval_accuracy_fine, eval_loss_coarse, eval_accuracy_coarse
- test_accuracy_fine (primary metric)

Checkpoints are saved to `<your-preferred-path>/checkpoints/<dataset>_<model_type>_<shots>shots/`.
Access WandB logs via your WandB dashboard.

> **Storage Considerations**:
> - Training checkpoints and WandB artifacts can consume significant storage space
> - It's recommended to periodically clean up old checkpoints and artifacts
> - Consider implementing a retention policy for checkpoints based on your storage constraints
> - For long-running experiments, monitor disk usage and adjust storage paths accordingly
