from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading images, labels, and metadata.
    """
    def __init__(self, img_paths, labels, metadata, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        try:
            image = Image.open(self.img_paths[index]).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {self.img_paths[index]}: {e}")
        label = self.labels[index]
        meta = self.metadata[index]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'metadata': meta}
        return sample        
    

       