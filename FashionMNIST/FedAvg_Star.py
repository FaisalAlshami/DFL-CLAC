import math
import os
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


from models import MNISTModelCNN
from non_iid import non_iid_distribution

import logging
from Utils import setup_logging, visualization_save_metrics, save_confusion_matrix, monitor_resources, logging_round_statics
from FL_Functions import train_model, evaluate_model, calculate_model_size


def _flatten_params(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a model's parameters into a single 1-D vector (cpu)."""
    return torch.cat([p.detach().cpu().view(-1) for p in state_dict.values()])


def _model_delta(global_sd: Dict[str, torch.Tensor], local_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return (local - global) parameter delta dict, on CPU."""
    delta = {}
    for k in global_sd:
        delta[k] = local_sd[k].detach().cpu() - global_sd[k].detach().cpu()
    return delta


def _add_state_dict(sd: Dict[str, torch.Tensor], add: Dict[str, torch.Tensor], alpha: float = 1.0):
    for k in sd:
        sd[k] = sd[k] + alpha * add[k].to(sd[k].device)


def _scale_state_dict(sd: Dict[str, torch.Tensor], scalar: float):
    for k in sd:
        sd[k] = sd[k] * scalar


def _zeros_like(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in sd.items()}


def _copy_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone().detach() for k, v in sd.items()}


def _trimmed_mean(stack: torch.Tensor, trim_ratio: float = 0.1) -> torch.Tensor:
    """Coordinate-wise trimmed mean. stack: [num_items, *param_shape]."""
    n = stack.shape[0]
    if n == 1:
        return stack[0]
    trim = int(max(0, math.floor(trim_ratio * n)))
    # reshape to [n, d]
    d = int(stack[0].numel())
    x = stack.reshape(n, d)
    # sort each column
    x_sorted, _ = torch.sort(x, dim=0)
    x_trimmed = x_sorted[trim : n - trim] if n - 2 * trim > 0 else x_sorted
    mean = x_trimmed.mean(dim=0)
    return mean.view_as(stack[0])


def _kmeans_numpy(X: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means on numpy. X: [N, D]. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N <= k:
        return np.arange(N), X.copy()
    # init centroids by random selection
    idx = rng.choice(N, size=k, replace=False)
    C = X[idx].copy()
    labels = np.zeros(N, dtype=np.int64)
    for _ in range(iters):
        # assign
        # squared euclidean distance
        dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        # update
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
    return labels, C




import random

def fashionmnist_fedavg_star(alpha=0.5, num_classes=10, num_clients=20, num_rounds=20, num_epochs=20,
                       batch_size=32, cluster_num=5, num_components=50, max_delay=60, selected_ratio=0.8, Allow_delay=60,
                       aggregation="FedAvg", topology="Star", dirName="FedAvg"):
    # Initialize the log file
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    log_filename = f"{topology}_{current_time}.log"

    # Prepare the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set up logging
    log_file = setup_logging(dirName, topology)
    logging.info(f"Starting Federated Learning with Star Topology. Log file: {log_file}")
    logging.info(f"This scenario to simulate FL with FedAvg algorithm according with following hyperparameters:\n"
                 f"*************** Dataset: CIFAR100,\n"
                 f"*************** num_clients: {num_clients},\n"
                 f"*************** alpha: {alpha}, \n"
                 f"*************** cluster_num:{cluster_num}, \n"
                 f"*************** num_rounds:{num_rounds}, \n"
                 f"*************** num_epochs: {num_epochs}, \n"
                 f"*************** batch_size:{batch_size}, \n"
                 f"*************** Aggregation Algorithm:{aggregation}, \n"
                 f"*************** topology:{topology}, \n"
                 f"*************** device:{device}, \n"
                 f"*************** Support spectral clustering: Yes, \n"
                 f"*************** Support Selecting supper client based on RL: No,\n "
                 f"*************** num_components(PCA):{num_components}")

    # Prepare MNIST dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std for grayscale
    ])

    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client_loaders, valid_client_indices = non_iid_distribution(num_clients, num_classes, train_data, batch_size)

    # Initialize the global model and start federated learning
    global_model = MNISTModelCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    # Log the start of training
    logging.info("Starting Federated Learning...\n")
    clients = [MNISTModelCNN().to(device) for _ in range(num_clients)]

    # Dictionary to store metrics for each round
    metrics_dict = {
        'Round': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [],
        'CPU Usage': [], 'RAM Usage': [], 'GPU Allocated': [], 'GPU Usage': [],
        'Round Duration (sec)': [],
        'Aggregation Duration (sec)': [],
        'Communication Overhead (bytes)': [],
        'Client Delays (sec)': [],
        'Count_Client_Delays': [],
        'Throughput': [], 'Straggler Rate': [], 'Staleness Impact': [], 'Effective Contribution Weight': [],
        'Client Participation Rate': [], 'Client Update Time': [], 'Cumulative Delay': []
    }

    # Federated learning communication rounds
    for round_num in range(num_rounds):
        round_start_time = time.time()
        logging.info(f"Communication Round {round_num + 1}/{num_rounds}")
        print(f"Communication Round {round_num + 1}/{num_rounds}")

        local_loss = 0
        local_accuracy = 0

        # Randomly select a subset of clients for this round
        selected_clients = valid_client_indices

        # Track delays and identify clients with delays > 1 minute
        client_delays = []
        clients_with_delays_over_1min = []  # Track clients with delays over 1 minute
        for client_idx in selected_clients:
            # Retrieve the corresponding DataLoader
            client_loader = client_loaders[valid_client_indices.index(client_idx)]

            local_model = MNISTModelCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())  # Start with global model

            local_optimizer = optim.Adam(local_model.parameters(), lr=0.001)

            # Train the local model for the specified number of epochs
            for epoch in range(num_epochs):
                local_loss, local_accuracy = train_model(local_model, client_loader, criterion, local_optimizer, device)

            #logging.info(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")
            #print(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")

            clients[client_idx].load_state_dict(local_model.state_dict())

            # Introduce random delay between 0 and 1 minute (in seconds)
            net_delay = random.uniform(0.0, max_delay)
            #time.sleep(net_delay)
            client_delays.append(net_delay)

            # # If delay is more than 1 minute, mark the client and exclude from aggregation
            # if net_delay > Allow_delay:
            #     clients_with_delays_over_1min.append(client_idx)

        # Exclude clients with delays > 1 minute from aggregation
        aggregation_clients = [client_idx for client_idx in selected_clients if client_idx not in clients_with_delays_over_1min]

        round_end_time = time.time()
        # FedAvg: Central server aggregates models from all clients
        aggregation_start_time = time.time()
        global_model_state_dict = global_model.state_dict()
        for key in global_model_state_dict:
            # Stack the parameters from all clients
            client_params = torch.stack(
                [clients[i].state_dict()[key].to(global_model_state_dict[key].device) for i in aggregation_clients])
            # Compute the mean of the parameters
            global_model_state_dict[key] = client_params.mean(0)
            # Load the averaged parameters back into the global model
        global_model.load_state_dict(global_model_state_dict)

        # Send the updated global model back to the clients
        for client_idx in aggregation_clients:  # Only send to clients that are not delayed
            clients[client_idx].load_state_dict(global_model.state_dict())

        global_model_size = calculate_model_size(global_model)
        print("Global Model Size: {:.2f} MB".format(global_model_size))

        # Evaluate the global model after aggregation
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(global_model, test_loader, device)

        # Round metrics
        #round_end_time = time.time()
        aggregation_end_time = time.time()
        round_duration = round_end_time - round_start_time
        aggregation_duration = aggregation_end_time - aggregation_start_time
        cumulative_delay = sum(client_delays)

        # Communication overhead: simple estimate based on the number of bytes sent
        comm_overhead = sum([len(clients[cid].state_dict()) for cid in aggregation_clients])

        # Throughput: Successful updates per minute
        throughput = len(aggregation_clients) / (round_duration / 60)  # updates per minute

        # Straggler rate: proportion of clients with excessive delays
        straggler_rate = sum([1 for delay in client_delays if delay > 30]) / len(client_delays)  # 30 seconds threshold for straggler

        # Metrics calculation
        client_participation_rate = len(aggregation_clients) / num_clients
        client_update_time = sum(client_delays) / len(aggregation_clients)

        # Store metrics in the dictionary
        metrics_dict['Round Duration (sec)'].append(round_duration)
        metrics_dict['Aggregation Duration (sec)'].append(aggregation_duration)
        metrics_dict['Communication Overhead (bytes)'].append(comm_overhead)
        metrics_dict['Client Delays (sec)'].append(sum(client_delays))
        metrics_dict['Throughput'].append(throughput)
        metrics_dict['Straggler Rate'].append(straggler_rate)
        metrics_dict['Staleness Impact'].append(np.mean(client_delays))  # Placeholder for staleness impact metric
        metrics_dict['Effective Contribution Weight'].append(np.sum(client_delays))  # Placeholder for effective contribution weight
        metrics_dict['Client Participation Rate'].append(client_participation_rate)
        metrics_dict['Client Update Time'].append(client_update_time)
        metrics_dict['Cumulative Delay'].append(cumulative_delay)
        metrics_dict['Count_Client_Delays'].append(len(clients_with_delays_over_1min))

        # Log results for each round
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Round {round_num + 1} - "
                     f"Global Accuracy: {test_accuracy:.4f}, "
                     f"Global Test Precision: {test_precision:.4f}, "
                     f"Global Test Recall: {test_recall:.4f}, "
                     f"Global F1Score: {test_f1:.4f}")

        # Monitor system resources
        cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv = monitor_resources()
        logging_round_statics(round_num, global_model_size, test_accuracy, test_f1, test_precision, test_recall,
                              cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv,
                              r_duration=round_duration,
                              client_delay=0,
                              throughput=throughput,
                              comm_overhead=comm_overhead
                              )

        # Save Confusion matrix
        predictions, true_labels = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.squeeze().cpu().numpy())
            true_labels.extend(target.cpu().numpy())

        # Store metrics
        metrics_dict["Round"].append(round_num + 1)
        metrics_dict["Accuracy"].append(test_accuracy)
        metrics_dict["Precision"].append(test_precision)
        metrics_dict["Recall"].append(test_recall)
        metrics_dict["F1 Score"].append(test_f1)
        metrics_dict["CPU Usage"].append(cpu_usage)
        metrics_dict["RAM Usage"].append(ram_usage)
        metrics_dict["GPU Allocated"].append(gpu_allocate)
        metrics_dict["GPU Usage"].append(gpu_usage_percent)
        logging.info(f"Save Confusion matrix for round {round_num + 1}")
        save_confusion_matrix(true_labels, predictions, round_num + 1, log_file)

        # Visualization and save metrics
        visualization_save_metrics(metrics_dict, log_file)

    # Save the final model after all rounds
    model_path = os.path.join(log_file, "global_model_fedavg_star.pth")
    torch.save(global_model.state_dict(), model_path)

if __name__ == "__main__":
    # Delay allows to clients Allow_delay
    fashionmnist_fedavg_star(alpha=0.5, num_classes=10, num_clients=20, num_rounds=10, num_epochs=20,
                        batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                        aggregation="FedAvg", topology="Star", dirName="FashionMNIST_FedAvg")


