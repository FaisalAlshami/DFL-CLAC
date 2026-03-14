# CNN Model for CIFAR-10 Dataset
import torch
from torch import nn

import torch.nn.functional as F


class CIFAR10CNNModel(nn.Module):
    def __init__(self):
        super(CIFAR10CNNModel, self).__init__()
        self.input_channels=3
        self.num_classes=10
        self.learning_rate=1e-3
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = self.learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.conv1 = torch.nn.Conv2d(self.input_channels, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer


import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNNModel1(nn.Module):
    def __init__(
            self,
            input_channels=1,  # Default: 1 for MNIST, can be adjusted for CIFAR10 (3 channels)
            num_classes=10,  # Number of output classes, e.g., 10 for MNIST
            learning_rate=1e-3  # Learning rate for Adam optimizer
    ):
        super(CIFAR10CNNModel1, self).__init__()

        # Store learning rate
        self.learning_rate = learning_rate

        # Define the layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,  # Input channels
            out_channels=32,  # Output channels
            kernel_size=3,  # 3x3 filter size
            padding=1  # Padding to keep the output size same
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,  # Input channels from previous layer
            out_channels=64,  # Output channels
            kernel_size=3,  # 3x3 filter size
            padding=1  # Padding to keep the output size same
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,  # Input channels from previous layer
            out_channels=128,  # Output channels
            kernel_size=3,  # 3x3 filter size
            padding=1  # Padding to keep the output size same
        )

        self.pool = nn.MaxPool2d(2, 2)  # MaxPool layer with 2x2 kernel, stride 2

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Input size is from the output of conv3
        self.fc2 = nn.Linear(1024, num_classes)  # Output size is the number of classes (10 for MNIST)

    def forward(self, x):
        # Apply convolutional layers with ReLU activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> ReLU -> pool
        x = self.pool(F.relu(self.conv3(x)))  # conv3 -> ReLU -> pool

        # Flatten the output of the last conv layer (for fully connected layers)
        x = x.view(-1, 128 * 4 * 4)  # Flatten to [batch_size, 128 * 4 * 4]

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer without ReLU (softmax is usually applied at inference)

        return x

    def configure_optimizers(self):
        # Use Adam optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MNISTModelCNN(nn.Module):
    def __init__(
            self,
            input_channels=1,
            num_classes=10,
            learning_rate=1e-3,
    ):
        super(MNISTModelCNN, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=2  # Explicit padding for 'same' behavior
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding=2  # Explicit padding for 'same' behavior
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Adjust for the flattened size after pooling
        self.l1 = nn.Linear(7 * 7 * 64, 2048)
        self.l2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # Apply conv1 and maxpool
        x = self.pool2(F.relu(self.conv2(x)))  # Apply conv2 and maxpool

        # Flatten the output of the last pool
        x = x.view(-1, 7 * 7 * 64)

        # Fully connected layers
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class CIFAR100ModelCNN1(nn.Module):
    def __init__(
            self,
            input_channels=3,      # CIFAR-100 images are RGB
            num_classes=100,       # 100 classes for CIFAR-100
            learning_rate=1e-3
    ):
        super(CIFAR100ModelCNN, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2  # same padding to keep feature map size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Image size: 32x32 → 16x16 → 8x8 after two poolings
        self.fc1 = nn.Linear(64 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # Output: 32×32 → 16×16
        x = self.pool2(F.relu(self.conv2(x)))  # Output: 16×16 → 8×8

        x = x.view(x.size(0), -1)  # Flatten: [batch, 64×8×8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class CIFAR100ModelCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100ModelCNN, self).__init__()

        # Define the layers of the model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input has 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

        self.fc1 = nn.Linear(256 * 2 * 2, 1024)  # Adjusted based on the final feature map size
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting

    def forward(self, x):
        # Apply convolutional layers with ReLU activations and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 2 * 2)  # Flatten to feed into fully connected layers

        # Apply the fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        # Adam optimizer with learning rate of 0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer