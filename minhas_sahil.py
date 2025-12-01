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

    model.to(device)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    with open(log_file, "w") as f:
        start_msg = f"Training {model.__class__.__name__}...\n"
        print(start_msg)
        f.write(start_msg)


        for epoch in range (1, epochs + 1):
            model.train()
            total_train_loss = 0

            # Go through batch of training images
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                # Reset gradients
                optimizer.zero_grad()
                # Make Predictions
                outputs = model(images)
                # Compute loss for batch
                loss = criterion(outputs, labels)
                # Gradients
                loss.backward()
                # Update model weights
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = total_val_loss / len(val_loader)

            val_acc = 100 * correct / total

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)


            printed_text = (
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%\n"
            )

            print(printed_text.strip())
            f.write(printed_text)
    torch.save(model.state_dict(), save_path)
    return train_losses, val_losses, val_accuracies


# 4. Evaluation Function  
def evaluate_model(model, test_loader, save_cm_path="confusion_matrix.jpg"):
    """Evaluate model performance and generate confusion matrix."""
    
    model.to(device)
    model.eval()

    predictions_lst = []
    labels_lst = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Make Predictions
            outputs = model(images)
            
            # Index of highest score to predicted digit
            _, predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions_lst.extend(predicted.cpu().numpy())
            labels_lst.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    cm = confusion_matrix(labels_lst, predictions_lst)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model.__class__.__name__} Confusion Matrix")
    plt.savefig(save_cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    return test_accuracy, cm


# 5. Visualization Function
def plot_curves(train_losses, val_losses, val_accuracies, model_name, save_path="curves.jpg"):
    """Plot and save training curves."""

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Loss curves graph according to example
    plt.subplot(1, 2, 1)  
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")
    plt.title(f"{model_name} - Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(False)

    # Accuracy curve graph according to example
    plt.subplot(1, 2, 2)  
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="green")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(False)

    
    plt.savefig(save_path, dpi=300)
    plt.close()


# 6. Main Execution
def main():
    """Main function to train and evaluate both models."""
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist_data()
    visualize_samples(train_loader.dataset, save_path="outputs/mnist_samples.png")

    # Train Feedforward NN
    nn_model = FeedforwardNN()

    nn_train_losses, nn_val_losses, nn_val_acc = train_model(
        model = nn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=0.001,
        save_path="outputs/nn_model.pth",
        log_file="outputs/nn_training_log.txt"
    )

    nn_test_acc, nn_cm = evaluate_model(
        model=nn_model,
        test_loader=test_loader,
        save_cm_path="outputs/nn_confusion_matrix.jpg"
    )

    plot_curves(
        nn_train_losses,
        nn_val_losses,
        nn_val_acc,
        model_name="Feedforward Neural Network",
        save_path="outputs/nn_training_curves.jpg"
    )

    # Train Convolutional NN
    cnn_model = CNN()

    cnn_train_losses, cnn_val_losses, cnn_val_acc = train_model(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=0.001,
        save_path="outputs/cnn_model.pth",
        log_file="outputs/cnn_training_log.txt"
    )

    cnn_test_acc, cnn_cm = evaluate_model(
        model=cnn_model,
        test_loader=test_loader,
        save_cm_path="outputs/cnn_confusion_matrix.jpg"
    )

    plot_curves(
    cnn_train_losses,
    cnn_val_losses,
    cnn_val_acc,
    model_name="Convolutional Neural Network",
    save_path="outputs/cnn_training_curves.jpg"
    )



    with open("outputs/results.txt", "w") as f:
        f.write("MNIST Classification Results Summary\n")
        f.write("========================================\n\n")

        # Feedforward NN summary
        f.write("Feedforward Neural Network:\n")
        f.write(f"- Test Accuracy: {nn_test_acc:.2f}%\n")
        f.write(f"- Final Validation Accuracy: {nn_val_acc[-1]:.2f}%\n")
        f.write(f"- Final Training Loss: {nn_train_losses[-1]:.4f}\n")
        f.write(f"- Final Validation Loss: {nn_val_losses[-1]:.4f}\n\n")

        # CNN summary
        f.write("Convolutional Neural Network:\n")
        f.write(f"- Test Accuracy: {cnn_test_acc:.2f}%\n")
        f.write(f"- Final Validation Accuracy: {cnn_val_acc[-1]:.2f}%\n")
        f.write(f"- Final Training Loss: {cnn_train_losses[-1]:.4f}\n")
        f.write(f"- Final Validation Loss: {cnn_val_losses[-1]:.4f}\n\n")

        # Comparison
        improvement = cnn_test_acc - nn_test_acc
        better_model = "CNN" if improvement > 0 else "Feedforward NN"

        f.write("Performance Comparison:\n")
        f.write(f"- CNN vs NN Accuracy Improvement: {improvement:.2f}%\n")
        f.write(f"- Better Model: {better_model}\n")

    # Print summary location
    print("\nTraining complete summary saved")
if __name__ == "__main__":
    main()