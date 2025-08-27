# Training Logic for Machine Learning Model

import os
import yaml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset  # Assuming CustomDataset is defined in dataset.py
from model import CustomModel  # Assuming CustomModel is defined in model.py

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    train_dataset = CustomDataset(config['data']['train_images'], config['data']['train_labels'], transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Model initialization
    model = CustomModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # Assuming a classification task
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {running_loss/len(train_loader):.4f}')

    # Save the trained model
    model_save_path = config['training']['model_save_path']
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    config = load_config('configs/default.yaml')
    train_model(config)