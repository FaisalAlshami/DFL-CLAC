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


from models import MNISTModelCNN, CIFAR10CNNModel
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



def cifar10_fedavg_star1(alpha=0.5, num_classes=10, num_clients=20, num_rounds=20, num_epochs=20,
                       batch_size=32, cluster_num=5, num_components=50, selected_ratio=0.8,
                       aggregation="FedAvg", topology="Star", dirName="FedAvg"):
    # Initialize the log file
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    log_filename = f"{topology}_{current_time}.log"

    # Prepare the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set up logging
    log_file = setup_logging(dirName,topology)
    logging.info(f"Starting Federated Learning with Star Topology. Log file: {log_file}")
    logging.info(f"This scenario to simulate FL with FedAvg algorithm according with following hyperparameters:\n"
                 f"*************** Dataset: CIFAR10,\n"
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

    # Step 1: Prepare CIFAR-10 dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std for grayscale
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client_loaders, valid_client_indices = non_iid_distribution(num_clients, num_classes, train_data, batch_size)




    # Step 3: Initialize the global model and start federated learning
    staleness_alpha = 0.25,
    global_model = CIFAR10CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    # Log the start of training
    logging.info("Starting Federated Learning...\n")
    # Initialize clients based on clustering labels
    clients = [CIFAR10CNNModel().to(device) for _ in range(num_clients)]

    # Dictionary to store metrics for each round
    metrics_dict = {
        "Round": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "CPU Usage": [],
        "RAM Usage": [],
        "GPU Allocated": [],
        "GPU Usage": [],
        "Round Duration (sec)":[],
        'Aggregation Duration (sec)': [],
        "Communication Overhead (bytes)":[],
        "Client Delays (sec)":[],
        "Throughput":[],
        "Straggler Rate": [],
        "Staleness Impact": [],
        "Effective Contribution Weight": [],
        "Client Participation Rate": [],
        "Client Update Time": [],
        "Cumulative Delay": [],
    }

    # Momentum buffer (velocity)
    v: Dict[str, torch.Tensor] = _zeros_like(global_model.state_dict())

    # Per-client delay stats (simulation) & participation weights
    last_finish_time = np.zeros(num_clients, dtype=np.float64)
    participation_weight = np.ones(num_clients, dtype=np.float64)

    # Federated learning communication rounds
    for round_num in range(num_rounds):
        round_start_time = time.time()
        logging.info(f"Communication Round {round_num + 1}/{num_rounds}")
        print(f"Communication Round {round_num + 1}/{num_rounds}")

        local_loss = 0
        local_accuracy = 0

        # Randomly select a subset of clients for this round
        selected_clients = valid_client_indices  # np.random.choice(valid_client_indices, size=int(selected_ratio * len(valid_client_indices)),replace=False)

        # Select subset of clients with probability ~ participation_weight
        m = max(1, int(selected_ratio * len(valid_client_indices)))
        weights = participation_weight[valid_client_indices]
        probs = weights / (weights.sum() + 1e-12)
        chosen = np.random.choice(valid_client_indices, size=m, replace=False, p=probs)

        # 2.1 Train selected clients locally & collect deltas
        global_sd_cpu = _copy_state_dict(global_model.state_dict())
        deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        flat_deltas: List[np.ndarray] = []
        delays: Dict[int, float] = {}

        start_round_time = time.time()

        # Track delays and client updates
        total_update_time = 0
        total_delay_time = 0
        straggler_count = 0

        # Track cumulative delay
        cumulative_delay = 0

        # Simulate the federated learning process for each selected client
        for client_idx in selected_clients:
            # Retrieve the corresponding DataLoader
            client_loader = client_loaders[valid_client_indices.index(client_idx)]

            local_model = CIFAR10CNNModel().to(device)
            local_model.load_state_dict(global_model.state_dict())  # Start with global model

            local_optimizer = optim.Adam(local_model.parameters(), lr=0.001)

            client_start = time.time()
            # Train the local model for the specified number of epochs
            for epoch in range(num_epochs):  # Train for 20 epochs
                local_loss, local_accuracy = train_model(local_model, client_loader, criterion, local_optimizer, device)


            # Log client-specific results
            #logging.info(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")
            #print(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")

            # Update the client model
            clients[client_idx].load_state_dict(local_model.state_dict())

            client_finish = time.time()

            # Simulate a random delay for each client (varying delays)
            net_delay = random.uniform(0.0, 0.1)  # seconds, random delay for each client
            #time.sleep(net_delay)
            finish_time = client_finish + net_delay
            last_finish_time[client_idx] = finish_time

            # Compute parameter delta
            local_sd = _copy_state_dict(local_model.state_dict())
            delta_sd = _model_delta(global_sd_cpu, local_sd)
            deltas[client_idx] = delta_sd
            flat_deltas.append(_flatten_params(delta_sd).numpy())

            delays[client_idx] = max(1e-6, finish_time - start_round_time)  # seconds since round start

            # Update client's local copy (optional)
            clients[client_idx].load_state_dict(local_sd)

            # Track update and delay times
            total_update_time += (client_finish - client_start)
            total_delay_time += net_delay

            # Count stragglers based on delay threshold
            if net_delay > 0.3:  # Consider clients with delay > 0.3 seconds as stragglers
                straggler_count += 1

            # Track cumulative delay
            cumulative_delay += net_delay


            # FedAvg: Central server aggregates models from all clients

        round_end_time = time.time()
        aggregation_start_time = time.time()
        # Normalize delays to [0,1] by dividing by max
        max_delay = max(delays.values()) if delays else 1.0
        norm_delay = {cid: delays[cid] / max_delay for cid in chosen}
        global_model_state_dict = global_model.state_dict()


        for key in global_model_state_dict:
            # Stack the parameters from all clients
            client_params = torch.stack(
                [clients[i].state_dict()[key].to(global_model_state_dict[key].device) for i in range(num_clients)])
            # Compute the mean of the parameters
            global_model_state_dict[key] = client_params.mean(0)
            # Load the averaged parameters back into the global model
        global_model.load_state_dict(global_model_state_dict)

        # Send the updated global model back to the clients
        for client_idx in range(num_clients):
            clients[client_idx].load_state_dict(global_model.state_dict())

        global_model_size = calculate_model_size(global_model)
        print("Global Model Size: {:.2f} MB".format(global_model_size))
        # Evaluate the global model after aggregation
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(global_model, test_loader, device)

        # Calculate Round Duration

        aggregation_end_time = time.time()
        r_duration = round_end_time - round_start_time
        aggregation_duration = aggregation_end_time - aggregation_start_time
        metrics_dict["Round Duration (sec)"].append(r_duration)
        metrics_dict['Aggregation Duration (sec)'].append(aggregation_duration)

        # Client delays
        client_delay = np.mean(list(delays.values()))
        metrics_dict["Client Delays (sec)"].append(client_delay)

        # Communication Overhead (simple estimate based on number of bytes sent)
        comm_overhead = sum([len(deltas[cid]) for cid in delays])
        metrics_dict["Communication Overhead (bytes)"].append(comm_overhead)

        # Throughput: Successful updates per minute
        throughput = m / (r_duration / 60)  # updates per minute
        metrics_dict["Throughput"].append(throughput)

        # Straggler Rate: proportion of clients with excessive delays
        straggler_rate = straggler_count / len(chosen)
        metrics_dict["Straggler Rate"].append(straggler_rate)

        # Staleness Impact: Average staleness factor (gamma)
        # staleness_impact = np.mean([math.exp(-staleness_alpha * delays[cid]) for cid in chosen])
        metrics_dict["Staleness Impact"].append(0)

        # Effective Contribution Weight: sum of weighted updates
        # effective_contribution_weight = np.sum([math.exp(-staleness_alpha * delays[cid]) for cid in chosen])
        metrics_dict["Effective Contribution Weight"].append(0)

        # Client Participation Rate: ratio of clients participating
        client_participation_rate = len(chosen) / num_clients
        metrics_dict["Client Participation Rate"].append(client_participation_rate)

        # Client Update Time: average time spent on client updates
        client_update_time = total_update_time / len(chosen)
        metrics_dict["Client Update Time"].append(client_update_time)

        # Cumulative Delay: sum of all client delays
        metrics_dict["Cumulative Delay"].append(cumulative_delay)

        # Log results for each round
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Round {round_num + 1} - "
                     f"Global Test set: Global Accuracy: {test_accuracy:.4f}, "
                     f"Global Test Precision: {test_precision:.4f}, "
                     f"Global Test Recall: {test_recall:.4f}, "
                     f"Global F1Score: {test_f1:.4f}")
        # Display system metrics
        # return cpu_usage, ram_usage, gpu_allocated, gpu_usage_percentage, bytes_sent, bytes_recv
        cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv = monitor_resources()
        logging_round_statics(round_num, global_model_size,
                              test_accuracy, test_f1, test_precision, test_recall,
                              cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv,
                              r_duration=r_duration,
                              client_delay=0,
                              throughput=throughput,
                              comm_overhead=comm_overhead
                              )

        predictions, true_labels = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.squeeze().cpu().numpy())
            true_labels.extend(target.cpu().numpy())

        logging.info(f"Save Confusion matrix for round {round_num + 1}")
        print(f"Save Confusion matrix for round {round_num + 1}")
        save_confusion_matrix(true_labels, predictions, round_num + 1, log_file)

        ##################
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
        # metrics_dict["Round Duration (sec)"].append(r_duration)
        # metrics_dict["Communication Overhead (bytes)"].append(comm_overhead)
        # metrics_dict["Throughput"].append(throughput)
        visualization_save_metrics(metrics_dict, log_file)


        ####################

    # Save the final model after all rounds
    model_path = os.path.join(log_file, "global_model_fedavg_star.pth")
    torch.save(global_model.state_dict(), model_path)

import random

def cifar10_fedavg_star(alpha=0.5, num_classes=10, num_clients=20, num_rounds=20, num_epochs=20,
                       batch_size=32, cluster_num=5, num_components=50, selected_ratio=0.8, Allow_delay=60,
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
                 f"*************** Dataset: CIFAR10,\n"
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

    # Prepare CIFAR-10 dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client_loaders, valid_client_indices = non_iid_distribution(num_clients, num_classes, train_data, batch_size)

    # Initialize the global model and start federated learning
    global_model = CIFAR10CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    # Log the start of training
    logging.info("Starting Federated Learning...\n")
    clients = [CIFAR10CNNModel().to(device) for _ in range(num_clients)]

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

            local_model = CIFAR10CNNModel().to(device)
            local_model.load_state_dict(global_model.state_dict())  # Start with global model

            local_optimizer = optim.Adam(local_model.parameters(), lr=0.001)

            # Train the local model for the specified number of epochs
            for epoch in range(num_epochs):
                local_loss, local_accuracy = train_model(local_model, client_loader, criterion, local_optimizer, device)

            #logging.info(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")
            #print(f"Client {client_idx + 1}: Loss={local_loss:.4f}, Local Accuracy={local_accuracy:.4f}")

            clients[client_idx].load_state_dict(local_model.state_dict())

            # Introduce random delay between 0 and 1 minute (in seconds)
            net_delay = random.uniform(0.0, 60.0)
            #time.sleep(net_delay)
            client_delays.append(net_delay)

            # If delay is more than 1 minute, mark the client and exclude from aggregation
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
    cifar10_fedavg_star(alpha=0.5, num_classes=10, num_clients=20, num_rounds=10, num_epochs=20,
                        batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                        aggregation="FedAvg", topology="Star", dirName="CIFAR10_FedAvg")

# if __name__ == "__main__":
#     #print(torch.__version__)  # Check the version
#     #print(torch.cuda.is_available())  # Should return True if CUDA is working
#
#
#    cifar10_fedavg_star(alpha=0.5,
#                        num_classes=10,
#                        num_clients=50,
#                        num_rounds=200,
#                        num_epochs=1,
#                        batch_size=128,
#                        cluster_num=6,
#                        num_components=100,
#                        selected_ratio=0.9,
#                        aggregation="FedAvg", #FedAvg or Wise
#                        topology="Star",
#                        dirName="CIFAR10_FedAvg")
