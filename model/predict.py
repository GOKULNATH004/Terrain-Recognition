"""Prediction script for terrain recognition"""

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from model import create_model
from data_loader import get_transforms


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = create_model(num_classes=NUM_CLASSES, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    return model


def predict_image(model, image_path, device):
    """Predict terrain class for single image"""
    # Load and transform image
    _, val_transform = get_transforms(INPUT_SIZE, augment=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    class_name = TERRAIN_CLASSES[predicted_class.item()]
    confidence = probabilities[0][predicted_class].item() * 100
    
    return class_name, confidence, probabilities[0].cpu().numpy()


def predict_batch(model, image_dir, device):
    """Predict terrain class for all images in directory"""
    results = []
    
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            class_name, confidence, probs = predict_image(model, img_path, device)
            results.append({
                'filename': img_name,
                'class': class_name,
                'confidence': confidence,
                'probabilities': probs
            })
            print(f"{img_name}: {class_name} ({confidence:.2f}%)")
    
    return results


def main():
    """Main prediction script"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Model not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Example: Predict on a single image
    print("\nTo use this script:")
    print("1. Call load_model(checkpoint_path, device) to load the model")
    print("2. Call predict_image(model, image_path, device) for single image prediction")
    print("3. Call predict_batch(model, image_dir, device) for batch prediction")


if __name__ == '__main__':
    main()
