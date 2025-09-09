# Leaf Disease Segmentation using U-Net

A deep learning project for semantic segmentation of leaf diseases using a custom U-Net architecture implemented in PyTorch.

## 🌿 Overview

This project implements a U-Net model for pixel-level segmentation of diseased areas in leaf images. The model can identify and segment 39 different classes of leaf diseases, making it useful for agricultural applications and plant health monitoring.

## 🏗️ Architecture

- **Model**: Custom U-Net with residual blocks and group normalization
- **Encoder**: Convolutional layers with downsampling
- **Bottleneck**: Residual blocks for feature processing
- **Decoder**: Upsampling with skip connections
- **Output**: 150-class semantic segmentation

### Key Features:
- Residual connections for better gradient flow
- Group normalization for stable training
- Configurable architecture with dimension multipliers
- Data augmentation with random crops and flips
- Mixed precision training support via Accelerate

## 📁 Project Structure

```
leaf-disease-segmentation-dataset-using-UNet/
├── UNetModel.py          # U-Net architecture implementation
├── dataloader.py         # Dataset class and data preprocessing
├── train.py             # Training script with Accelerate
├── show_result.py       # Visualization of model predictions
└── leaf-disease-segmentation-dataset/
    ├── data/            # Original dataset
    │   ├── images/      # Input leaf images
    │   └── annotations/ # Segmentation masks
    └── aug_data/        # Augmented dataset (optional)
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install accelerate transformers
pip install matplotlib pillow numpy tqdm
pip install torchsummary
```

### Dataset Setup
1. Download the leaf disease segmentation dataset
2. Place images in `data/images/` and annotations in `data/annotations/`
3. Run the dataloader to automatically split into train/validation sets
4. DataSet canbe access for kagle

```Python
import kagglehub
path = kagglehub.dataset_download("fakhrealam9537/leaf-disease-segmentation-dataset")
print("Path to dataset files:", path)
```

### Training
```python
python train.py
```

### Inference & Visualization
```python
python show_result.py
```

## 🔧 Configuration

### Model Parameters
- **Input channels**: 3 (RGB)
- **Output classes**: 39 (disease categories)
- **Image size**: 128x128 (configurable)
- **Architecture**: Configurable depth with `dim_mults=[1,2,4]`

### Training Parameters
- **Batch size**: 4 (with gradient accumulation)
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss

## 📊 Features

### Data Augmentation
- Random resized crops (20%-100% of original size)
- Random horizontal flips
- Normalization with ImageNet statistics

### Training Features
- Gradient accumulation for effective larger batch sizes
- Gradient clipping for stable training
- Automatic checkpoint saving
- Progress tracking with tqdm
- Mixed precision training support

## 🎯 Results

The model outputs pixel-wise predictions for 150 different disease classes. Use `show_result.py` to visualize:
- Original input images
- Ground truth segmentation masks
- Model predictions

## 📈 Model Architecture Details

### Residual Blocks
- Group normalization for better convergence
- SiLU activation function
- Skip connections for gradient flow

### U-Net Structure
- **Encoder**: Progressive downsampling with feature extraction
- **Bottleneck**: Deep feature processing
- **Decoder**: Upsampling with skip connections from encoder
- **Skip Connections**: Preserve spatial information across scales

## 🛠️ Customization

### Modify Architecture
```python
model = UNet(
    in_channels=3,
    num_classes=39,
    start_dim=32,           # Starting feature dimension
    dim_mults=[1,2,4,8],    # Dimension multipliers per level
    residual_block_per_group=2,  # Blocks per resolution
    groupnorm_num_group=8,  # Group norm groups
    interpolate_upsample=True    # Use interpolation vs transpose conv
)
```

### Adjust Training
```python
train(
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    image_size=256
)
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- Original U-Net paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Implementation inspired by PyTorch Adventures tutorials