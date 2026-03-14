"""Training script for terrain recognition model"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from model import create_model
from data_loader import get_data_loaders


def setup_device():
    """Setup device (CUDA if available, CPU otherwise)"""
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        DEVICE = torch.device('cpu')
        print("CUDA not available. Using CPU")
    return DEVICE


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = 100 * correct / total
    
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = 100 * correct / total
    
    return avg_loss, avg_accuracy


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch


def main():
    """Main training loop"""
    print("=" * 60)
    print("Terrain Recognition Model Training")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Create model
    print(f"\nCreating model with {NUM_CLASSES} classes...")
    model = create_model(num_classes=NUM_CLASSES, device=device)
    print(f"Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    print(f"\nLoading dataset from {DATASET_PATH}...")
    print(f"Classes: {TERRAIN_CLASSES}")
    
    try:
        train_loader, val_loader, dataset = get_data_loaders(
            dataset_path=DATASET_PATH,
            class_names=TERRAIN_CLASSES,
            batch_size=BATCH_SIZE,
            input_size=INPUT_SIZE,
            augment=USE_AUGMENTATION,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure dataset directory exists at: {os.path.abspath(DATASET_PATH)}")
        return
    
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, EPOCHS, val_loss, final_path)


if __name__ == '__main__':
    main()
