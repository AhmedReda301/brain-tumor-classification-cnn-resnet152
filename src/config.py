import torch
from pathlib import Path

# Device config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 3
SEED = 20

# Split dataset output path
SPLIT_DATA_DIR = Path(r'D:\Projects\DL\Brian Tumor\Data\Brain Tumor Data Set')

# Metadata CSV path
METADATA_PATH = Path(r'D:\Projects\DL\Brian Tumor\Data\metadata.csv')


