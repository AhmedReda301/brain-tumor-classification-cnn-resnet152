import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from config import SPLIT_DATA_DIR, BATCH_SIZE, METADATA_PATH, SEED
from custom_data import CustomDataset
from glob import glob
import os

def load_image_paths_and_labels(base_path):
    image_paths = []
    labels = []

    for class_name in ['Brain Tumor', 'Healthy']:
        class_folder = os.path.join(base_path, class_name)
        class_label = 1 if class_name == 'Brain Tumor' else 0

        # Read all images inside class_folder
        for img_path in glob(os.path.join(class_folder, '*')):
            image_paths.append(img_path)
            labels.append(class_label)
    
    return image_paths, labels

# Load metadata (example assumes CSV with one row per image)
def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')  # List of dicts, one per sample

transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)

def get_dataloaders():
    # Load image paths and labels
    image_paths, labels = load_image_paths_and_labels(SPLIT_DATA_DIR)
    metadata = load_metadata(METADATA_PATH)

    # Split into train and val sets (80% train, 20% val)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Optionally, also split metadata if it matters
    train_metadata = [m for p, m in zip(image_paths, metadata) if p in train_paths]
    val_metadata = [m for p, m in zip(image_paths, metadata) if p in val_paths]

    # Create datasets
    train_dataset = CustomDataset(train_paths, train_labels, train_metadata, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, val_metadata, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader
