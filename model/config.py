"""Configuration settings for terrain recognition model"""

# Dataset paths
DATASET_PATH = "../dataset"
TERRAIN_CLASSES = ["Desert", "Forest", "Mountain", "Plains"]
NUM_CLASSES = len(TERRAIN_CLASSES)

# Model parameters
INPUT_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Checkpoint and logging
CHECKPOINT_DIR = "./checkpoints"
LOG_INTERVAL = 10
SAVE_INTERVAL = 5  # Save checkpoint every N epochs

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Device (will be auto-set to cuda if available, else cpu)
DEVICE = None  # Will be set in code based on availability

# Data augmentation
USE_AUGMENTATION = True
