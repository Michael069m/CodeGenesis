#!/bin/bash

# Setup script for ML Training Project
# Run this script on your friend's laptop after cloning the repository

set -e  # Exit on any error

echo "🚀 Setting up ML Training Project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment and install dependencies
echo "📥 Installing dependencies..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    .venv/Scripts/activate
    .venv/Scripts/pip install --upgrade pip
    .venv/Scripts/pip install -r requirements.txt
else
    # macOS/Linux
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Add your dataset to data/train/ and data/val/ directories"
echo "2. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   .venv\\Scripts\\activate"
else
    echo "   source .venv/bin/activate"
fi
echo "3. Start training:"
echo "   .venv/bin/yolo detect train data=configs/yolo_dataset.yaml model=yolov8n.pt epochs=50"
echo ""
echo "📊 Monitor training progress in: runs/train/"
