# StreamSplit: Theoretical Guarantees for Edge Audio Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **StreamSplit: Theoretical Guarantees for Edge Audio Learning** (AAAI 2026).

## Overview

StreamSplit enables efficient audio representation learning on edge devices through three key innovations:

1. **Streaming Contrastive Learning** - Distribution-based learning with convergence guarantees
2. **Hybrid Loss Function** - Laplacian regularization + Sliced-Wasserstein distance
3. **RL-Guided Computation Splitting** - PPO-based adaptive workload division

### Key Results
- 77.1% bandwidth reduction, 72.6% latency reduction, 52.3% energy savings
- Within 2.2% accuracy of server-only baseline
- Training: 4.01 → 2.33 loss (41.8% improvement, 100 epochs)

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Edge Device    │◄───────►│  Server          │
│  - Audio FFT    │         │  - Aggregation   │
│  - Local Loss   │         │  - Hybrid Loss   │
│  - Memory Bank  │         │  - Refinement    │
│  - RL Agent     │         │  - Prototypes    │
└─────────────────┘         └──────────────────┘
```

**Core Innovation**: PPO-based RL agent (`edge/rl_splitting.py`, 609 lines) dynamically selects optimal split points (layers 0,2,4,7,11,13) in MobileNetV3 based on resource constraints and network conditions.

## Installation

```bash
git clone https://github.com/yourusername/StreamSplit-AAAI.git
cd StreamSplit-AAAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Prepare datasets
python scripts/download_audioset.py
python scripts/prepare_edge_data.py

# 2. Train the system (100 epochs, ~2h on CPU)
python train.py --config configs/streamsplit.yaml

# 3. Run tests
python demo.py  # All 12 tests should pass
```

**Training Results**: Loss reduces from 4.01 → 2.33 over 100 epochs (41.8% improvement). Checkpoints saved every 10 epochs to `./checkpoints/` (162MB each).

**Pre-trained Models**: Download from [Zenodo](https://zenodo.org/PLACEHOLDER) or [Hugging Face](https://huggingface.co/PLACEHOLDER)

## Project Structure

```
StreamSplit-AAAI/
├── edge/                      # Edge device modules
│   ├── rl_splitting.py        # PPO-based split controller (609 lines)
│   ├── contrastive_learning.py# Distribution-based loss
│   ├── memory_bank.py         # Prototype memory
│   └── audio_processing.py    # FFT, MFCC
├── server/                    # Server modules
│   ├── hybrid_loss.py         # Laplacian + Sliced-Wasserstein
│   └── aggregation.py         # Multi-device aggregation
├── models/
│   └── mobilenet_v3.py        # Encoder with split points
├── datasets/
│   ├── audioset.py
│   └── edge_audio.py
├── train.py                   # Main training script
├── demo.py                    # Component tests
└── configs/
    └── streamsplit.yaml       # Hyperparameters
```

## Configuration

Key hyperparameters in `configs/streamsplit.yaml`:

```yaml
training:
  num_epochs: 100

edge:
  batch_size: 200
  learning_rate: 0.0001  # 1e-4
  fft_window_ms: 25
  temperature: 0.1
  feature_dim: 128
  memory_bank_size: [64, 512]

server:
  batch_size: 256
  learning_rate: 0.0005  # 5e-4
  lambda_laplacian: 0.5
  num_projections: 100

rl:
  algorithm: "ppo"
  gamma: 0.99
  hidden_dim: 128
  momentum: 0.999
```

## Hardware Requirements

- **Edge**: Raspberry Pi 4B (4GB RAM) or equivalent
- **Server**: 8+ cores, 32GB RAM, NVIDIA GPU (optional)
- **Tested on**: macOS (Apple Silicon), Ubuntu 20.04/22.04, Raspberry Pi 4B

## Citation

```bibtex
@inproceedings{streamsplit2026,
  title={StreamSplit: Theoretical Guarantees for Edge Audio Learning},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built on PyTorch, AudioSet, MobileNetV3, and PPO. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
