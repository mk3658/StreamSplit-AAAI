# Quick Start Guide

This guide will help you get started with StreamSplit in 5 minutes.

## 1. Installation (2 minutes)

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/StreamSplit-AAAI.git
cd StreamSplit-AAAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python demo.py
```

You should see all demos passing ✓

## 2. Run a Quick Demo (1 minute)

Test the core components:

```python
import torch
from models import MobileNetV3Small

# Create model
model = MobileNetV3Small(width_mult=0.75, embedding_dim=128)

# Generate dummy audio features (mel-spectrogram)
x = torch.randn(4, 1, 128, 100)  # [batch, channel, mels, time]

# Get embeddings
embeddings = model(x)
print(f"Embeddings shape: {embeddings.shape}")  # [4, 128]
```

## 3. Understanding the Architecture (1 minute)

StreamSplit has three main components:

### Edge Device
```python
from edge import AudioProcessor, MemoryBank, StreamingContrastiveLearning

# Process audio on edge
processor = AudioProcessor(config)
mel_spec = processor.process_audio(waveform)

# Store in memory bank
memory_bank = MemoryBank(min_size=64, max_size=512, embedding_dim=128)
```

### Server
```python
from server import HybridLoss, UncertaintyEstimator

# Hybrid loss for refinement
loss_fn = HybridLoss(config)
loss = loss_fn(edge_embeddings, server_embeddings)
```

### Metrics & Visualization
```python
from utils import plot_tsne, MetricsTracker

# Track metrics
tracker = MetricsTracker()
tracker.update({'loss': 0.5, 'accuracy': 0.85})

# Visualize embeddings
plot_tsne(embeddings, labels, save_path='tsne.png')
```

## 4. Configuration (30 seconds)

Edit `configs/streamsplit.yaml`:

```yaml
edge:
  memory_bank:
    min_size: 64
    max_size: 512
  
server:
  hybrid_loss:
    lambda_laplacian: 0.5
    num_projections: 100
```

## 5. Training (30 seconds to start)

```bash
# Start training (requires dataset)
python train.py --config configs/streamsplit.yaml
```

Note: You'll need to implement dataset loaders first. See `train.py` for TODOs.

## Common Tasks

### Process Audio
```python
from edge import AudioProcessor, AudioAugmentor
import torch

# Load config
import yaml
with open('configs/streamsplit.yaml') as f:
    config = yaml.safe_load(f)

# Create processor
processor = AudioProcessor(config)
augmentor = AudioAugmentor(config)

# Process audio
waveform = torch.randn(1, 16000 * 10)  # 10 seconds at 16kHz
mel_spec = processor.process_audio(waveform)
waveform_aug = augmentor.augment(waveform)
```

### Monitor Resources
```python
from edge import ResourceMonitor

monitor = ResourceMonitor(config)
monitor.start()

# Get current state
state = monitor.get_state()
print(f"CPU: {state['cpu_util']*100:.1f}%")
print(f"Memory: {state['mem_usage']*100:.1f}%")

monitor.stop()
```

### Compute Embeddings
```python
from models import MobileNetV3Small
import torch

model = MobileNetV3Small()
model.eval()

with torch.no_grad():
    embeddings = model(mel_spectrogram)
```

### Evaluate Metrics
```python
from utils.metrics import (compute_retrieval_precision_at_k,
                          compute_silhouette_score)

# Retrieval precision
precision = compute_retrieval_precision_at_k(embeddings, labels, k=10)

# Clustering quality
silhouette = compute_silhouette_score(embeddings, labels)
```

## Next Steps

1. **Prepare your data**: Implement dataset loaders in a new `datasets/` module
2. **Train a model**: Run `train.py` with your dataset
3. **Evaluate**: Use the metrics in `utils/metrics.py`
4. **Visualize**: Generate plots with `utils/visualization.py`
5. **Deploy**: Test on Raspberry Pi with `requirements-edge.txt`

## Troubleshooting

### Import errors
```bash
# Make sure you're in the right directory
cd StreamSplit-AAAI

# Reinstall
pip install -e .
```

### Resource monitoring not working
```bash
# Install psutil
pip install psutil
```

### CUDA errors
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if needed
# Edit config: device: "cpu"
```

## Getting Help

- **Demo not working?** Run `python demo.py` to see specific errors
- **Questions?** Open an issue on GitHub
- **Contributing?** See `CONTRIBUTING.md`

## Resources

- [Full README](README.md) - Complete documentation
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details
- [Citation](CITATION.md) - How to cite this work
- [Paper PDFs](main.pdf, appendix.pdf) - Original research

---

**Ready to start?** Run `python demo.py` now! 🚀
