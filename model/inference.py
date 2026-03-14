"""Simple utility for quick inference on terrain images"""

import os
import sys
import torch
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from predict import load_model, predict_image


class TerrainPredictor:
    """Simple interface for terrain prediction"""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to model checkpoint (uses best_model.pth if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = load_model(model_path, self.device)
        self.class_names = TERRAIN_CLASSES
    
    def predict(self, image_path):
        """
        Predict terrain type for an image
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: {
                'class': terrain_type,
                'confidence': float (0-100),
                'probabilities': {class_name: probability, ...}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        class_name, confidence, probs = predict_image(
            self.model, image_path, self.device
        )
        
        prob_dict = {
            name: float(prob) for name, prob in zip(self.class_names, probs)
        }
        
        return {
            'class': class_name,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }
    
    def predict_batch(self, image_dir):
        """
        Predict terrain type for all images in directory
        
        Args:
            image_dir: Path to directory containing images
        
        Returns:
            list: List of prediction results
        """
        results = []
        
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"Directory not found: {image_dir}")
        
        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                try:
                    result = self.predict(img_path)
                    result['filename'] = img_name
                    results.append(result)
                    print(f"✓ {img_name}: {result['class']} ({result['confidence']:.2f}%)")
                except Exception as e:
                    print(f"✗ {img_name}: Error - {e}")
        
        return results


def main():
    """Example usage"""
    print("Terrain Recognition Predictor")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = TerrainPredictor()
        print("✓ Model loaded successfully")
        print(f"  Device: {predictor.device}")
        print()
        
        # Example single image prediction
        # Uncomment and modify path to test
        # result = predictor.predict('path/to/image.jpg')
        # print(f"Prediction: {result['class']} ({result['confidence']:.2f}%)")
        # print(f"All probabilities: {json.dumps(result['probabilities'], indent=2)}")
        
        print("Usage:")
        print("-" * 50)
        print("Initialize:")
        print("  predictor = TerrainPredictor()")
        print()
        print("Predict single image:")
        print("  result = predictor.predict('path/to/image.jpg')")
        print("  print(f\"{result['class']}: {result['confidence']:.2f}%\")")
        print()
        print("Predict batch:")
        print("  results = predictor.predict_batch('path/to/images/')")
        print()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python train.py")


if __name__ == '__main__':
    main()
