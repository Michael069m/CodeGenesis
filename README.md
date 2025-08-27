# ML Training Project - YOLOv8 Object Detection

This project is designed for training YOLOv8 object detection models on custom datasets. It includes scripts for data preparation, model training, evaluation, and testing with a focus on entity box detection.

## Project Overview

- **Model**: YOLOv8n (nano) for fast object detection
- **Dataset**: Custom YOLO format with images and bounding box labels
- **Task**: Single-class object detection (entity_box)
- **Framework**: Ultralytics YOLOv8

## Quick Start 

### 1. Clone the Repository

```bash
git clone https://github.com/Michael069m/CodeGenesis.git
cd CodeGenesis
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Data

```bash
# Copy your images and labels to the data directories
# Training images: data/train/images/
# Training labels: data/train/labels/
# Validation images: data/val/images/
# Validation labels: data/val/labels/

# If you need to split training data, use:
python scripts/split_train_val.py --data-dir data --val-ratio 0.1

# Normalize label filenames if needed:
python scripts/normalize_yolo_labels.py --images-dir data/train/images --labels-dir data/train/labels
python scripts/normalize_yolo_labels.py --images-dir data/val/images --labels-dir data/val/labels
```

### 4. Start Training

```bash
# Smoke test (1 epoch, quick validation):
.venv/bin/yolo detect train data=configs/yolo_dataset.yaml model=yolov8n.pt imgsz=640 epochs=1 batch=8 workers=2 device=auto project=runs/train name=smoke_test

# Full training (50 epochs):
.venv/bin/yolo detect train data=configs/yolo_dataset.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16 workers=4 device=auto project=runs/train name=full_training

# Monitor training progress in: runs/train/[experiment_name]/
```

### 5. Evaluate Results

```bash
# Check training results
ls runs/train/full_training/

# View training plots and metrics
# Open: runs/train/full_training/results.png
# Best weights: runs/train/full_training/weights/best.pt
```

## Project Structure

```
CodeGenesis
├── .venv/                   # Virtual environment (created during setup)
├── configs/
│   └── yolo_dataset.yaml    # YOLO dataset configuration
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── labels/          # Training labels (.txt files)
│   ├── val/
│   │   ├── images/          # Validation images
│   │   └── labels/          # Validation labels (.txt files)
│   └── test/
│       └── golden_test_set/ # Test dataset
├── scripts/
│   ├── split_train_val.py   # Split data into train/val
│   └── normalize_yolo_labels.py # Normalize label filenames
├── runs/                    # Training outputs (created automatically)
│   └── train/              # Training experiment results
├── src/                     # Custom training code (optional)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

├── requirements.txt # Python dependencies required for the project
├── pyproject.toml # Project configuration
├── .gitignore # Files and directories to be ignored by Git
└── README.md # Documentation for the project

```

## Setup Instructions

1. Clone the repository:
```

git clone <repository-url>
cd ml-training-project

```

2. Install the required dependencies:
```

pip install -r requirements.txt

```

3. Prepare the dataset:
- Place your training images in the `data/train/images` directory.
- Place your training labels in the `data/train/labels` directory.
- Follow the scripts in the `scripts` folder for data preparation and splitting.

4. Train the model:
```

python src/train.py

```

5. Evaluate the model:
```

python src/evaluate.py

```

## Usage Guidelines

- Modify the `configs/default.yaml` file to adjust training parameters such as learning rate and batch size.
- Use the Jupyter notebook in the `notebooks` folder for exploratory data analysis.
- Ensure to run tests in the `tests` folder to validate your implementation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
