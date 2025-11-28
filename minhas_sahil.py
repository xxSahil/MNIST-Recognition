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
        #MNIST images are 28x28 pixels
        input_size = 28 * 28

        # Hidden layers
        hidden_layer1 = 256
        hidden_layer2 = 128

        # Output size is number of classes
        output_size = 10

        # Flatten to a 1D vector for fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, output_size)

        # Activation Function
        self.relu = nn.ReLU()
        
        # Helps reduce overfitting
        self.dropout = nn.Dropout(p = 0.2)
    
    def forward(self, x):
        # Flatten to a 1D vector
        x = self.flatten(x)

        # First hidden layer + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second hidden layer + ReLU + Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)
        
        return x


class CNN(nn.Module):
    """Convolutional Neural Network for MNIST classification."""
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution layers; Layer 1 (1 input, 32 output), Layer 2 (32 input, 64 output)
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32, 
            kernel_size = 3
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3
        )

        # Pooling layer reduces size by half
        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        
        # Dropout to help reduce overfitting
        self.dropout = nn.Dropout(p = 0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()

    
    def forward(self, x):
        # Convolution 1 + ReLU + Pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolution 2 + ReLU + Pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # Fully connected layer + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

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