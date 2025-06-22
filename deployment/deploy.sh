#!/bin/bash
# deploy.sh - Build and deploy the fruit classifier

set -e

echo "🍌 Fruit Classifier Deployment Script"
echo "===================================="

# Configuration
MODEL_NAME="fruit-classifier"
VERSION=${1:-latest}
MODEL_DIR="../data/06_models/automm_fruit_classifier"
# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found at $MODEL_DIR"
    echo "Please train the model first with: kedro run"
    exit 1
fi

# Save label map alongside model
echo "📝 Extracting label map..."
python -c "
import json
from pathlib import Path

# Your label map from training
label_map = {
    '0': 'Banana 1',
    '1': 'Strawberry 1',
    '2': 'Watermelon 1'
}

with open('${MODEL_DIR}/../label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
print('✅ Label map saved')
"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t ${MODEL_NAME}:${VERSION} .

# Option 1: Run with docker-compose
echo "🚀 Starting with docker-compose..."
docker-compose up -d

echo "✅ Deployment complete!"
echo ""
echo "📍 API available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "Test with:"
echo "  curl -X POST -F 'file=@path/to/fruit.jpg' http://localhost:8000/predict"
