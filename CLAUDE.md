# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is focused on building custom Vision-Language-Action (VLA) models based on the LeRobot framework. The main goals include:
- Extending LeRobot with new environments and datasets
- Implementing novel VLA architectures and training methods
- Integrating reference implementations from UniVLA, OpenPI, and other state-of-the-art approaches
- Creating a unified framework for embodied intelligence research

## Commands

### LeRobot Core Commands
```bash
# Train policies using LeRobot framework
lerobot-train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-train --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht
lerobot-train --policy.type=smolvla --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Train with custom environments and datasets (for VLA development)
lerobot-train --policy.type=custom_vla --env.type=custom_env --dataset.repo_id=local/custom_dataset
lerobot-train --config_path=custom_configs/vla_config.json

# Evaluate trained policies
lerobot-eval --policy.path=path/to/checkpoints/pretrained_model --env.type=aloha
lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht

# Reproduce SOTA results from hub
lerobot-train --config_path=lerobot/diffusion_pusht

# Visualize datasets (including custom datasets)
python -m src.lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0
python -m src.lerobot.scripts.visualize_dataset --repo-id local/custom_dataset --episode-index 0 --local-files-only
```

### Testing Commands
```bash
# Run all end-to-end tests
make test-end-to-end

# Run specific policy tests
make test-act-ete-train
make test-diffusion-ete-train
make test-smolvla-ete-train

# Run unit tests
python -m pytest tests/
python -m pytest tests/test_policies.py
```

### Development and Setup Commands
```bash
# Install LeRobot in editable mode (required for custom development)
pip install -e .

# Install with specific simulation environments
pip install -e ".[aloha,pusht,xarm]"

# Install additional dependencies for VLA development
pip install -e ".[transformers-dep,pi0,smolvla]"

# GPU occupation script (prevents GPU conflicts during development)
python occ.py --devices="0,1,2,3"

# Setup wandb for experiment tracking
wandb login

# Find cameras and motors (for real robot data collection)
lerobot-find-cameras
lerobot-find-port
```

### Custom VLA Development Commands
```bash
# Create new environment implementations
python -c "from src.lerobot.envs import register_env; register_env('custom_env', CustomEnvClass)"

# Create custom datasets from local data
python scripts/convert_dataset.py --input_dir /path/to/raw_data --output_dir ./custom_datasets

# Test custom policy implementations
python -m pytest tests/test_custom_policies.py

# Benchmark custom models against baselines
python scripts/benchmark_policies.py --custom_policy custom_vla --baseline_policy act

# Direct access to LeRobot modules for development
python -c "import sys; sys.path.append('./src'); from lerobot.policies import CustomVLAPolicy"
```

### Data Management Commands
```bash
# Record new datasets with custom environments
lerobot-record --env.type=custom_env --save_dir ./custom_datasets

# Record with specific configurations for VLA training
lerobot-record --env.type=aloha --episodes=100 --episode_length=1000

# Replay existing datasets
lerobot-replay --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Replay custom local datasets
lerobot-replay --dataset.repo_id=local/custom_dataset --local-files-only

# Process and upload custom datasets to Hugging Face Hub
python -m src.lerobot.scripts.push_dataset_to_hub --dataset_dir ./custom_datasets
```

## Architecture Overview

This repository serves as a development environment for custom VLA (Vision-Language-Action) models built on top of the LeRobot framework. It combines the robustness of LeRobot with cutting-edge VLA research from reference implementations.

### Key Components

**LeRobot Framework (in working directory):**
- **Core Library**: State-of-the-art machine learning for real-world robotics in PyTorch
- **Extensible Policies (`src/lerobot/policies/`)**: 
  - ACT (Action Chunking Transformer): For imitation learning tasks
  - Diffusion Policy: For complex manipulation behaviors
  - SMOLVLA: Vision-language-action model for instruction following
  - TDMPC: Model-predictive control for continuous control
  - VQ-BeT: Vector-quantized behavior transformers
  - PI0: Flow matching models for continuous action prediction
  - **Custom VLA Policies**: Direct modification and extension space

- **Extensible Datasets (`src/lerobot/datasets/`)**: 
  - LeRobot dataset format with video compression and efficient loading
  - Support for ALOHA, PushT, XArm environments
  - Automatic download from Hugging Face Hub
  - Statistics caching for fast startup
  - **Custom Dataset Support**: Direct integration of new data sources
  - **Multi-modal Data**: Vision-language-action triplets support

- **Flexible Environments (`src/lerobot/envs/`)**: 
  - Base simulation environments (ALOHA, PushT, XArm)
  - Real robot interfaces (ALOHA hardware, various motor controllers)
  - Evaluation frameworks with success metrics
  - **Custom Environment Registration**: Direct implementation space
  - **Language-Conditioned Tasks**: Instruction-following scenarios

- **Scripts (`src/lerobot/scripts/`)**: 
  - Training and evaluation pipelines
  - Dataset visualization tools
  - Model conversion utilities
  - **Direct Modification**: Easy customization of training loops

**VLA Reference Implementations (`reference/`):**
- **UniVLA**: Vision-language-action models with multimodal architectures
  - Multimodal fusion techniques for vision and language
  - Large-scale pre-training approaches
  - Fine-tuning strategies for robot tasks
- **OpenPI**: Open vocabulary robot manipulation policies
  - Language-conditioned policy learning
  - Zero-shot generalization capabilities
- **Bagel**: Large-scale behavior modeling approaches
  - Scalable training methodologies
  - Multi-task learning frameworks
- **Chain-of-Action**: Sequential action prediction frameworks
  - Hierarchical action decomposition
  - Long-horizon planning capabilities

**Development Areas for Custom VLA:**
- **Novel Architectures**: Experiment with new multimodal fusion methods
- **Training Strategies**: Implement custom loss functions and optimization techniques
- **Environment Extensions**: Create domain-specific robotic environments
- **Dataset Augmentation**: Develop data collection and synthetic data generation pipelines

### Data Pipeline Architecture

1. **LeRobot Dataset Format**: Unified format with video compression for efficient storage
2. **Hugging Face Hub Integration**: Automatic dataset download and caching
3. **Multi-modal Support**: Images, robot states, actions with temporal relationships
4. **Statistics Caching**: Pre-computed dataset statistics for fast loading
5. **Transform Pipeline**: Configurable image transforms and data augmentation

### Training Architecture

1. **Policy-Agnostic Framework**: Supports multiple policy types (ACT, Diffusion, SmolVLA, etc.)
2. **Distributed Training**: Multi-GPU support with automatic device detection
3. **Configuration System**: Hydra-based configs for reproducible experiments
4. **Checkpointing**: Automatic model saving with resume capability
5. **Evaluation**: Environment-specific evaluation with success metrics
6. **Experiment Tracking**: Weights & Biases integration for monitoring

### GPU Management

The repository includes cooperative GPU sharing:
- `occ.py`: Multi-GPU occupation script with intelligent yielding
- Automatic detection of other users via GPU utilization monitoring
- Graceful memory management and cleanup
- Support for distributed training coordination

### Key Features for VLA Development

- **Modular Architecture**: Easy to extend and customize for new VLA approaches
- **Pre-trained Models**: Hub-hosted models ready for evaluation and fine-tuning
- **Custom Environment Support**: Framework for implementing domain-specific robotic tasks
- **Language Integration**: Built-in support for instruction-following and language conditioning
- **Multi-modal Learning**: Vision-language-action learning pipelines
- **Simulation Environments**: ALOHA, PushT, XArm for training and testing
- **Real Robot Support**: Hardware interfaces for physical deployment
- **Visualization Tools**: Dataset and training visualization via Rerun
- **End-to-End Testing**: Comprehensive test suite with CI/CD integration
- **Reference Integration**: Easy access to state-of-the-art VLA implementations

## Important Paths and Configuration

- **Working Directory**: `/opt/tiger/univla/` (repository root with LeRobot integrated)
- **LeRobot Source**: `src/lerobot/` (main framework code for direct modification)
- **LeRobot Policies**: `src/lerobot/policies/` (add custom VLA policies here)
- **LeRobot Environments**: `src/lerobot/envs/` (extend with custom environments)
- **LeRobot Datasets**: `src/lerobot/datasets/` (integrate custom data formats)
- **LeRobot Scripts**: `src/lerobot/scripts/` (customize training/eval pipelines)
- **Reference Code**: `reference/` (VLA implementations for study and adaptation)
- **GPU Management**: `occ.py` (cooperative GPU sharing utility)
- **Python Package**: `pyproject.toml` (dependencies and project configuration)
- **Default Cache**: `~/.cache/huggingface/lerobot/` (dataset storage)
- **Custom Datasets**: `./custom_datasets/` (local dataset development)
- **Experiment Outputs**: Configurable via `--output_dir` in training commands
- **Test Suite**: `tests/` (LeRobot test framework for validation)

## Development Workflow for Custom VLA

1. **Study Reference Implementations**: Analyze UniVLA, OpenPI, etc. in `reference/`
2. **Extend LeRobot Framework**: Add custom policies directly in `src/lerobot/policies/`
3. **Create Custom Environments**: Implement domain-specific tasks in `src/lerobot/envs/`
4. **Integrate Custom Datasets**: Add data processing in `src/lerobot/datasets/`
5. **Customize Training Scripts**: Modify `src/lerobot/scripts/` for specific needs
6. **Collect/Process Data**: Use data management commands for custom datasets
7. **Train and Evaluate**: Use LeRobot's training infrastructure with custom components
8. **Test and Validate**: Run tests in `tests/` directory for reliability
9. **Iterate and Improve**: Leverage visualization and testing tools for development

## Quick Start for VLA Development

```bash
# 1. Install in development mode
pip install -e .

# 2. Study existing policies
ls src/lerobot/policies/

# 3. Create your custom policy
cp src/lerobot/policies/act/modeling_act.py src/lerobot/policies/custom_vla/

# 4. Register your policy
# Edit src/lerobot/policies/__init__.py to include your custom policy

# 5. Test your implementation
python -m pytest tests/test_policies.py::test_custom_vla

# 6. Train your custom VLA
lerobot-train --policy.type=custom_vla --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Common VLA Development Patterns

- **Multimodal Fusion**: Combine vision and language encoders for action prediction
- **Instruction Following**: Condition policies on natural language commands  
- **Transfer Learning**: Fine-tune pre-trained models on custom datasets
- **Hierarchical Planning**: Decompose complex tasks into sub-goals and actions
- **Sim-to-Real Transfer**: Train in simulation and deploy on real robots