# Integration tests for the training pipeline

import os
import pytest
from src.train import train_model
from src.dataset import CustomDataset

@pytest.fixture
def setup_data():
    # Setup paths for training and validation datasets
    train_images_path = os.path.join('data', 'train', 'images')
    train_labels_path = os.path.join('data', 'train', 'labels')
    val_images_path = os.path.join('data', 'val', 'images')
    val_labels_path = os.path.join('data', 'val', 'labels')

    # Create dataset instances
    train_dataset = CustomDataset(train_images_path, train_labels_path)
    val_dataset = CustomDataset(val_images_path, val_labels_path)

    return train_dataset, val_dataset

def test_training_pipeline(setup_data):
    train_dataset, val_dataset = setup_data

    # Train the model
    model = train_model(train_dataset)

    # Evaluate the model on the validation dataset
    metrics = model.evaluate(val_dataset)

    # Check if the metrics are as expected
    assert metrics['accuracy'] > 0.7  # Example threshold
    assert metrics['loss'] < 0.5  # Example threshold

def test_data_integrity(setup_data):
    train_dataset, _ = setup_data

    # Check if the dataset has the expected number of images
    assert len(train_dataset) == 10000  # Assuming 10,000 images for training

    # Check if images and labels are correctly paired
    for img, label in train_dataset:
        assert os.path.exists(img)
        assert os.path.exists(label)