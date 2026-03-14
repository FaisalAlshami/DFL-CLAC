import logging
from datetime import datetime
import os
import time

import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


# Train the model for one epoch
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


# Evaluate the model on the test set
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Convert logits to class predictions

            all_preds.extend(predicted.cpu().numpy())  # Convert to CPU for F1 score
            all_labels.extend(labels.cpu().numpy())  # Convert to CPU for F1 score

    # Calculate Accuracy and F1 Score

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)  # 'weighted' for multi-class
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)  # 'weighted' for multi-class
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)  # Use 'weighted' for multi-class

    return accuracy, precision, recall, f1




def calculate_model_size(model):
    # Calculate the size of the model in bytes
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_in_mb = model_size / (1024 ** 2)  # Convert bytes to megabytes
    logging.info(f"Global Size Model: {model_size_in_mb:.2f} MB\n")
    print(f"Global Size Model: {model_size_in_mb:.2f} MB\n")
    return model_size_in_mb

