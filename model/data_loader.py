"""Data loading utilities for terrain recognition dataset"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class TerrainDataset(Dataset):
    """Custom dataset for terrain images"""
    
    def __init__(self, root_dir, class_names, transform=None):
        """
        Args:
            root_dir: Path to dataset directory
            class_names: List of class names
            transform: Transformation to apply to images
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        self._load_images()
    
    def _load_images(self):
        """Load all images from dataset"""
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(size=224, augment=True):
    """
    Get image transformations
    
    Args:
        size: Input image size
        augment: Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def get_data_loaders(dataset_path, class_names, batch_size=32, 
                     input_size=224, augment=True, num_workers=2):
    """
    Create train and validation data loaders
    
    Args:
        dataset_path: Path to dataset directory
        class_names: List of class names
        batch_size: Batch size for loaders
        input_size: Input image size
        augment: Whether to apply augmentation
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    train_transform, val_transform = get_transforms(input_size, augment)
    
    # Create dataset
    dataset = TerrainDataset(dataset_path, class_names, transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset = type('obj', (object,), {
        'root_dir': dataset.root_dir,
        'class_names': dataset.class_names,
        'transform': val_transform,
        'images': dataset.images,
        'labels': dataset.labels,
        '__len__': dataset.__len__,
        '__getitem__': lambda self, idx: (
            val_transform(Image.open(dataset.images[idx]).convert('RGB')),
            dataset.labels[idx]
        )
    })()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, dataset
