#!/usr/bin/env python3
# test_client.py - Test the fruit classifier API

import requests
import sys
from pathlib import Path

def test_prediction(image_path: str, api_url: str = "http://localhost:8000"):
    """Test single image prediction"""
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    # Prepare the file
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        
        # Make request
        response = requests.post(f"{api_url}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Image: {image_path}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   All probabilities:")
        for fruit, prob in result['probabilities'].items():
            print(f"     - {fruit}: {prob:.2%}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")

def test_health(api_url: str = "http://localhost:8000"):
    """Test API health"""
    response = requests.get(f"{api_url}/")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ API Status: {result['status']}")
        print(f"   Model loaded: {result['model_loaded']}")
    else:
        print(f"❌ API unavailable")

if __name__ == "__main__":
    print("🍓 Fruit Classifier API Test")
    print("============================")
    
    # Test health
    test_health()
    print()
    
    # Test predictions
    if len(sys.argv) > 1:
        for image_path in sys.argv[1:]:
            test_prediction(image_path)
            print()
    else:
        print("Usage: python test_client.py <image1.jpg> [image2.jpg ...]")
        print("\nExample test images:")
        print("  python test_client.py data/01_raw/Banana\\ 1/*.jpg")