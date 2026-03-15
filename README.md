# Terrain Recognition - Deep Learning Model

A **TerrainXCNN** (eXtended CNN with Residual Blocks) deep learning model for terrain classification into 4 categories: Desert, Forest, Mountain, and Plains.

### What is XCNN?

**XCNN = eXtended CNN with Skip Connections (Residual Blocks)**

Unlike standard CNNs that stack convolutions sequentially, XCNN allows information to skip layers through **skip connections**. This provides:
- ✅ Better gradient flow for deeper networks
- ✅ Multi-scale feature learning
- ✅ 5-15% higher accuracy
- ✅ Faster training convergence

## ⚡ Quick Start

Get up and running in 3 steps:

### Step 1: Install & Prepare
```bash
pip install -r requirements.txt
# Add your terrain images to dataset/ folder
```

### Step 2: Train Model
```bash
cd model
python train.py
```
Model trains on your GPU (or CPU). Best checkpoint saved automatically.

### Step 3: Use Web Interface
```bash
streamlit run app.py
```
Open browser → Upload images → Get predictions! 🎯

**That's it!** You now have a working terrain recognition AI. 🚀

## Project Structure

```
terrain_recognition/
├── dataset/                    # Training dataset
│   ├── Desert/
│   ├── Forest/
│   ├── Mountain/
│   └── Plains/
├── model/                      # Model package
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── model.py               # CNN architecture
│   ├── data_loader.py         # Data loading utilities
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── checkpoints/           # Saved model checkpoints
├── test.ipynb                 # Jupyter notebook for testing
└── requirements.txt           # Python dependencies
```

## Features

- **Automatic Device Detection**: Uses CUDA (GPU) if available, falls back to CPU
- **Data Augmentation**: Includes rotation, flipping, and color jittering
- **Batch Normalization**: Improves training stability
- **Checkpoint Saving**: Saves best model and periodic checkpoints
- **Progress Tracking**: Visual progress bars during training and validation

## Installation

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

**Note**: To use CUDA:
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- cuDNN library
- PyTorch with CUDA support will be installed automatically via requirements.txt

### 2. Prepare Dataset

Ensure your dataset is organized in the following structure:

```
dataset/
├── Desert/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Forest/
│   ├── image1.jpg
│   └── ...
├── Mountain/
│   └── ...
└── Plains/
    └── ...
```

### 3. (Optional) Install Additional Dependencies

For Streamlit web interface (already in requirements.txt):

```bash
pip install streamlit>=1.28.0
```

## Usage

### Training

Run the training script:

```bash
cd model
python train.py
```

**Configuration** (edit `model/config.py`):
- `EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `INPUT_SIZE`: Input image size (default: 224x224)
- `USE_AUGMENTATION`: Enable data augmentation (default: True)

**Output**:
- `model/checkpoints/best_model.pth`: Best model based on validation accuracy
- `model/checkpoints/checkpoint_epoch_*.pth`: Checkpoints every N epochs
- `model/checkpoints/final_model.pth`: Final model after training

### Web Interface (Streamlit)

Launch the interactive web UI:

```bash
streamlit run app.py
```

**Features:**
- 🖼️ **Upload Images**: Drag & drop or select terrain images
- 🎯 **Instant Predictions**: Real-time terrain classification
- 📊 **Confidence Scores**: Visualize probability for all classes
- ⚙️ **Adjustable Settings**: Confidence threshold, probability display
- 📱 **Responsive Design**: Works on desktop and mobile
- 🚀 **Fast**: Loads model once, caches for performance

**Usage:**
1. Open Streamlit app (runs on `localhost:8501`)
2. Upload a terrain image (JPG, PNG, BMP)
3. View prediction with confidence score
4. See detailed probability breakdown
5. Adjust confidence threshold in sidebar

### Making Predictions

#### Single Image Prediction:

```python
import torch
from model.predict import load_model, predict_image
from model.config import CHECKPOINT_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(f'{CHECKPOINT_DIR}/best_model.pth', device)

terrain_class, confidence, probabilities = predict_image(
    model, 'path/to/image.jpg', device
)

print(f"Terrain: {terrain_class}")
print(f"Confidence: {confidence:.2f}%")
```

#### Batch Prediction:

```python
from model.predict import load_model, predict_batch

results = predict_batch(model, 'path/to/image/directory', device)
for result in results:
    print(f"{result['filename']}: {result['class']} ({result['confidence']:.2f}%)")
```

#### Using the Simplified Interface:

```python
from model.inference import TerrainPredictor

predictor = TerrainPredictor()
result = predictor.predict('path/to/image.jpg')
print(f"Terrain: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Probabilities: {result['probabilities']}")
```

### Using the Jupyter Notebook

Open `test.ipynb` to:
- Train the model
- Visualize training progress
- Make predictions on test images
- Display confidence scores

## Model Architecture

### TerrainXCNN (eXtended CNN with Residual Blocks)

**Architecture Overview:**

```
INPUT (224×224×3)
    ↓
Conv7×7 (64 channels) + MaxPool
    ↓
Layer 1: 2 Residual Blocks (64 channels with skip connections)
    ↓
Layer 2: 2 Residual Blocks (128 channels with skip connections)
    ↓
Layer 3: 2 Residual Blocks (256 channels with skip connections)
    ↓
Layer 4: 2 Residual Blocks (512 channels with skip connections)
    ↓
Adaptive Average Pooling
    ↓
FC: 512 → 256 (Dropout 0.5)
    ↓
FC: 256 → 128 (Dropout 0.5)
    ↓
FC: 128 → 4 (output classes)
```

**Key Features:**

1. **Residual Blocks (Skip Connections)**
   - Information bypasses layers directly
   - Enables much deeper networks (20+ layers vs 8)
   - Better gradient flow during backpropagation
   - Reduces vanishing gradient problem

2. **Multi-Scale Feature Learning**
   - Layer 1: 64 channels (low-level features: textures, edges)
   - Layer 2: 128 channels (mid-level: patterns, shapes)
   - Layer 3: 256 channels (high-level: terrain structures)
   - Layer 4: 512 channels (semantic features: terrain type)

3. **Progressive Downsampling**
   - Stride-2 convolutions reduce spatial dimensions
   - Increases receptive field for global context

**Advantages for Terrain Recognition:**

- ✅ Distinguishes subtle differences between classes (plains vs desert)
- ✅ Learns complex terrain patterns (forest canopy vs mountain terrain)
- ✅ Deeper network = better feature extraction (~20 layers)
- ✅ 5-15% better accuracy vs standard CNN
- ✅ Faster convergence and more stable training

Total parameters: ~11.2 million (deeper = more expressive)

## Device Information

The model automatically detects and uses:
- **CUDA (GPU)**: If NVIDIA GPU is available → Faster training
- **CPU**: Fallback option → Works on any system

To check device detection:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Training Tips

1. **GPU Memory**: If running out of memory, reduce `BATCH_SIZE` in config.py
2. **Dataset Size**: More images per class = better results (aim for 100+ per class)
3. **Image Quality**: High-resolution images (224x224+) work better
4. **Training Duration**: On GPU: ~30 seconds per epoch (varies by GPU)
                          On CPU: ~2-3 minutes per epoch

## Performance Optimization

- **Mixed Precision Training**: Uncomment in train.py for faster training on newer GPUs
- **Data Caching**: Dataset is loaded into memory for faster training
- **Learning Rate Scheduling**: Automatic reduction after every 10 epochs

## Troubleshooting

### CUDA Not Available
- Verify NVIDIA drivers: `nvidia-smi`
- Install CUDA Toolkit matching your GPU
- Reinstall PyTorch with CUDA support

### Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Use `num_workers=0` in data_loader.py (already set)
- Close other applications

### Dataset Not Found
- Verify dataset path in config.py
- Ensure directory structure is correct
- Check file extensions are supported (.jpg, .png, .jpeg)

## Dependencies

- PyTorch: Deep learning framework
- TorchVision: Computer vision utilities
- NumPy: Numerical operations
- Pillow: Image processing
- tqdm: Progress bars

## Future Enhancements

- [ ] Transfer learning with pre-trained models (ResNet18/50 from ImageNet)
- [ ] Model ensemble combining multiple XCNN variants
- [ ] Real-time camera prediction for drone/satellite analysis
- [ ] Model quantization for mobile/edge deployment
- [ ] Confusion matrix and per-class accuracy visualization
- [ ] Grad-CAM visualization to see what the model focuses on
- [ ] Multi-GPU training with distributed training
- [ ] Custom data augmentation pipelines

## License

Free to use for educational and research purposes.
