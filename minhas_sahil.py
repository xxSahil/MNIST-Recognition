import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data Preprocessing
def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training set into training and validation (80:20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

def visualize_samples(dataset, save_path="mnist_samples.png"):
    """Visualize one sample per digit (0-9)."""
    samples = {i: None for i in range(10)}
    for img, label in dataset:
        if samples[label] is None:
            samples[label] = img
        if all(v is not None for v in samples.values()):
            break
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.set_title(f'Digit {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 2. Model Definitions
class FeedforwardNN(nn.Module):
    """Feedforward Neural Network for MNIST classification."""
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        # Your code here - Design your feedforward neural network architecture
        # Consider: input size, hidden layers, output size, activation functions
        pass
    
    def forward(self, x):
        # Your code here - Implement forward pass
        pass

class CNN(nn.Module):
    """Convolutional Neural Network for MNIST classification."""
    def __init__(self):
        super(CNN, self).__init__()
        # Your code here - Design your CNN architecture
        # Consider: convolutional layers, pooling, fully connected layers
        pass
    
    def forward(self, x):
        # Your code here - Implement forward pass
        pass

# 3. Training Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, save_path="model.pth", log_file="training_log.txt"):
    """Train a model and log performance metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Your code here - Implement training loop with validation and logging
    # Requirements:
    # - Train for specified epochs
    # - Track and log training/validation loss and accuracy
    # - Save training progress to log_file in the specified format
    # - Save trained model to save_path
    # - Return training metrics for plotting
    
    pass

# 4. Evaluation Function  
def evaluate_model(model, test_loader, save_cm_path="confusion_matrix.jpg"):
    """Evaluate model performance and generate confusion matrix."""
    # Your code here - Implement model evaluation
    # Requirements:
    # - Calculate test accuracy
    # - Generate and save confusion matrix as JPEG
    # - Return accuracy and confusion matrix
    
    pass

# 5. Visualization Function
def plot_curves(train_losses, val_losses, val_accuracies, model_name, save_path="curves.jpg"):
    """Plot and save training curves."""
    # Your code here - Create and save training/validation curves
    # Requirements:
    # - Plot training and validation loss curves
    # - Plot validation accuracy curve
    # - Save as high-quality JPEG
    
    pass

# 6. Main Execution
def main():
    """Main function to train and evaluate both models."""
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist_data()
    visualize_samples(train_loader.dataset, save_path="outputs/mnist_samples.png")
    
    # Your code here - Train and evaluate both models
    # Requirements:
    # - Train FeedforwardNN and CNN models
    # - Generate all required outputs (logs, confusion matrices, curves)
    # - Save results summary comparing both models
    
    pass

if __name__ == "__main__":
    main()