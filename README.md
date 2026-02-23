# PhysX-MKS-GhostNet

A physics-informed deep learning framework for SAR (Synthetic Aperture Radar) target recognition using the MSTAR dataset. This project combines GhostNet architecture with physics-based constraints and multi-kernel scattering principles for robust target classification.

## 🎯 Overview

This repository implements a novel approach to SAR Automatic Target Recognition (ATR) by integrating:
- **PhysX-MKS Architecture**: Physics-informed GhostNet with Multi-Kernel Scattering
- **Physics-based Loss Functions**: Incorporating electromagnetic scattering principles
- **Advanced Data Augmentation**: Complex-valued transformations for SAR imagery
- **Extended Operating Conditions (EOC)**: Robust performance across depression angles and configurations

## 📁 Project Structure

```
PhysX-MKS-GhostNet/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   └── net_architecture.py   # PhysX-MKS-GhostNet implementation
│   ├── dataset.py                # MSTAR dataset loader
│   ├── losses.py                 # Physics-informed loss functions
│   ├── transforms.py             # SAR-specific augmentations
│   └── utils.py                  # Utility functions
├── deployment/                   # Deployment configurations
│   ├── quantize.py              # Model quantization to INT8
│   └── pi_inference.py          # Raspberry Pi inference benchmarking
├── outputs/                      # Training outputs and results
│   ├── physx_ghost.onnx         # Exported ONNX model
│   ├── physx_ghost_int8.onnx    # Quantized ONNX model
│   ├── results/                  # Prediction results (numpy arrays)
│   ├── tables/                   # Performance tables
│   └── visualizations/           # Generated figures and plots
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── fine_tune.py                  # Fine-tuning script
├── test_with_tta.py             # Test-time augmentation
├── export_to_onnx.py            # Model export to ONNX
├── environment.yml              # Conda environment
└── req.txt                      # Python requirements
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- MSTAR dataset

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mannava-Daasaradhi/PhysX-MKS-GhostNet.git
cd PhysX-MKS-GhostNet
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate physx_ghost
```

Or using pip:
```bash
pip install -r req.txt
```

### Training

```bash
python train.py
```

Configuration parameters in `train.py`:
- `BATCH_SIZE`: 16
- `EPOCHS`: 100
- `LEARNING_RATE`: 1e-3
- `WEIGHT_DECAY`: 1e-3
- `LABEL_SMOOTHING`: 0.1

### Evaluation

```bash
# Standard evaluation
python evaluate.py

# Evaluation with Test-Time Augmentation
python test_with_tta.py
```

### Fine-tuning

```bash
python fine_tune.py
```

## 📊 Features

### Data Augmentation
- **ComplexRandomScale**: Scale transformations for complex SAR data
- **ComplexRandomRotation**: Rotation-invariant feature learning
- **ComplexSpeckleNoise**: Realistic SAR noise simulation
- **RandomPhaseShift**: Phase perturbations
- **ComplexRandomErasing**: Occlusion and variant simulation

### Model Architecture
- **GhostNet Backbone**: Efficient feature extraction
- **Physics-Informed Constraints**: Electromagnetic scattering principles
- **Multi-Kernel Scattering**: Enhanced feature representation
- **Reconstruction Branch**: Self-supervised physics constraints

### Loss Function
PhysXLoss combining:
- Cross-entropy classification loss with label smoothing
- Reconstruction loss for physics consistency
- Scattering consistency loss
- Gradient clipping for stable training

## 📈 Results

The model achieves >91% accuracy on Extended Operating Conditions (EOC) tasks.

Pre-computed results are available in the `outputs/` directory:
- **outputs/results/**: Prediction arrays for analysis
- **outputs/tables/**: Performance tables
- **outputs/visualizations/**: Figures and plots

## 🔧 Deployment

### Export to ONNX
```bash
python export_to_onnx.py
```

### Quantization (INT8)
```bash
python deployment/quantize.py
```

### Raspberry Pi Inference
```bash
python deployment/pi_inference.py --model outputs/physx_ghost.onnx
```

Benchmark different thread configurations for optimal performance on edge devices.

## 📦 Dependencies

Core dependencies:
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

See `req.txt` or `environment.yml` for complete dependencies.

## 📝 MSTAR Dataset

This project uses the MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset for SAR target classification.

Target classes are defined in `src/dataset.py` as `MSTAR_CLASSES`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add your license here]

## 📧 Contact

Mannava Daasaradhi - [@Mannava-Daasaradhi](https://github.com/Mannava-Daasaradhi)

## 🙏 Acknowledgments

- MSTAR dataset providers
- GhostNet architecture authors
- Physics-informed neural networks research community

---

**Note**: This is a research project. Pre-computed results are included in `outputs/` for reproducibility.
