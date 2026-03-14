import os
import time
import math
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models import MNISTModelCNN
from non_iid import non_iid_distribution

import logging
from Utils import (
    setup_logging,
    save_confusion_matrix,
    monitor_resources,
    visualization_save_metrics_dacey, visualization_save_metrics,
)
from FL_Functions import train_model, evaluate_model, calculate_model_size

# -----------------------------
# DFL-CLAC components (helpers)
# -----------------------------

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


def logging_round_statics(round_num,
                          global_model_size,
                          test_accuracy,
                          test_f1,
                          test_precision,
                          test_recall,
                          cpu_usage,
                          ram_usage,
                          gpu_allocate,
                          gpu_usage_percent,
                          bytes_sent,
                          bytes_recv,
                          r_duration:0,
                          client_delay:0,
                          throughput:0,
                          comm_overhead:0,):
    # "Round Duration (sec)": [],
    # "Client Delays (sec)": [],
    # "Throughput": [],
    # "Communication Overhead (bytes)": [],
    print(f"System Metrics - CPU: {cpu_usage}%, RAM: {ram_usage} MB, GPU Allocated: {gpu_allocate} MB, GPU Usage:{gpu_usage_percent}%")
    print(f"Network Metrics - MB Sent: {bytes_sent}, MB Received: {bytes_recv}")
    logging.info(f"System Metrics - CPU: {cpu_usage}%, RAM: {ram_usage}MB, GPU Allocated: {gpu_allocate} MB, GPU Usage:{gpu_usage_percent}%")
    logging.info(f"Network Metrics - MB Sent: {bytes_sent}, MB Received: {bytes_recv}")
    print(f" Global Model Size: {global_model_size:.2f} MB")
    logging.info(f" Global Model Size: {global_model_size:.2f} MB")
    # Print results for debugging
    logging.info(f"Round {round_num + 1}: Accuracy = {test_accuracy:.4f}, "
                 f"F1 Score = {test_f1:.4f}, "
                 f"Precision = {test_precision:.4f}, "
                 f"Recall = {test_recall:.4f}")
    print(f"Round {round_num + 1}: Accuracy = {test_accuracy:.4f}, "
          f"F1 Score = {test_f1:.4f}, "
          f"Precision = {test_precision:.4f}, "
          f"Recall = {test_recall:.4f}")


    logging.info(f"Round {round_num + 1}: r_duration = {r_duration:.4f}, "
                 f"throughput = {throughput:.4f}, "
                 f"comm_overhead = {comm_overhead:.4f}")


    print(f"Round {round_num + 1}: r_duration = {r_duration:.4f}, "
                 f"throughput = {throughput:.4f}, "
                 f"comm_overhead = {comm_overhead:.4f}")

    logging.info("-" * 50 + "\n")


# --------------------------------------
# Main: CIFAR10 with DFL-CLAC (Star Topo)
# --------------------------------------

def fashionmnist_dfl_clac_star(
    alpha: float = 0.5,
    num_classes: int = 100,
    num_clients: int = 20,
    num_rounds: int = 20,
    num_epochs: int = 5,
    batch_size: int = 32,
    k_clusters: int = 5,
    selected_ratio: float = 0.8,

    # CLAC / Robust aggregation
    trim_ratio: float = 0.1,  # for inter-cluster trimmed-mean

    # Staleness & momentum
    staleness_alpha: float = 0.05,  # gamma = exp(-alpha * delay)
    momentum_beta: float = 0.9,     # v <- beta v + update
    momentum_lr: float = 1.0,       # w <- w + momentum_lr * v

    # Reclustering / stragglers
    recluster_every: int = 1,       # re-cluster each round by default
    straggler_theta: float = 0.75,  # if normalized delay > theta -> straggler
    max_delay=60,
    topology: str = "Star",
    dirName: str = "FashionMNIST_DFL_CLAC",
):
    """
    Simulate DFL-CLAC on FashionMNIST under a Star topology.
    Key features implemented:
      • Adaptive clustering of clients by per-round update directions (k-means)
      • Layered aggregation: intra-cluster mean -> inter-cluster trimmed-mean
      • Time-sensitive staleness weighting: gamma = exp(-alpha * delay)
      • Momentum smoothing on the global update trajectory
      • Straggler detection and adaptive sampling (reduce selection prob for stragglers)
    """

    # Initialize the log file
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    _ = f"{topology}_{current_time}.log"

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set up logging
    log_file = setup_logging(dirName, topology)
    logging.info(f"Starting DFL-CLAC (Star). Log dir: {log_file}")
    logging.info(
        "\n".join([
            "Scenario: FashionMNIST with DFL-CLAC",
            f"num_clients={num_clients}",
            f"alpha={alpha}",
            f"k_clusters={k_clusters}",
            f"num_rounds={num_rounds}",
            f"num_epochs={num_epochs}",
            f"batch_size={batch_size}",
            f"trim_ratio={trim_ratio}",
            f"staleness_alpha={staleness_alpha}",
            f"momentum_beta={momentum_beta}",
            f"momentum_lr={momentum_lr}",
            f"recluster_every={recluster_every}",
            f"straggler_theta={straggler_theta}",
            f"topology={topology}",
            f"device={device}",
        ])
    )

    # ---------------------------
    # 1) Data: MNIST loaders
    # ---------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std for grayscale
    ])
    train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client_loaders, valid_client_indices = non_iid_distribution(num_clients, num_classes, train_data, batch_size)

    # ---------------------------
    # 2) Initialize models
    # ---------------------------
    global_model = MNISTModelCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    clients: List[nn.Module] = [MNISTModelCNN().to(device) for _ in range(num_clients)]
    for c in clients:
        c.load_state_dict(global_model.state_dict())

    # Momentum buffer (velocity)
    v: Dict[str, torch.Tensor] = _zeros_like(global_model.state_dict())

    # Per-client delay stats (simulation) & participation weights
    last_finish_time = np.zeros(num_clients, dtype=np.float64)
    participation_weight = np.ones(num_clients, dtype=np.float64)

    # Metrics storage
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

    # ---------------------------
    # Federated rounds
    # ---------------------------
    for r in range(num_rounds):
        round_start_time = time.time()
        logging.info(f"\n===== Round {r+1}/{num_rounds} =====")
        print(f"Round {r+1}/{num_rounds}")

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

        for cid in chosen:
            client_model = MNISTModelCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(client_model.parameters(), lr=0.001)

            # Simulate variable compute/comm latency by random sleep proportional to data size
            # (In real systems, measure wall-clock; here, we emulate.)
            client_start = time.time()

            for _ in range(num_epochs):
                loader = client_loaders[valid_client_indices.index(cid)]
                _loss, _acc = train_model(client_model, loader, criterion, optimizer, device)

            client_finish = time.time()

            # Simulate a random delay for each client (varying delays)
            net_delay = random.uniform(0.0, max_delay,)  # seconds, random delay for each client
            #time.sleep(net_delay)
            finish_time = client_finish + net_delay
            last_finish_time[cid] = finish_time

            # Compute parameter delta
            local_sd = _copy_state_dict(client_model.state_dict())
            delta_sd = _model_delta(global_sd_cpu, local_sd)
            deltas[cid] = delta_sd
            flat_deltas.append(_flatten_params(delta_sd).numpy())

            delays[cid] = max(1e-6, finish_time - start_round_time)  # seconds since round start

            # Update client's local copy (optional)
            clients[cid].load_state_dict(local_sd)

            # Track update and delay times
            total_update_time += (client_finish - client_start)
            total_delay_time += net_delay

            # Count stragglers based on delay threshold
            if net_delay > 0.3:  # Consider clients with delay > 0.3 seconds as stragglers
                straggler_count += 1

            # Track cumulative delay
            cumulative_delay += net_delay

        # Normalize delays to [0,1] by dividing by max
        max_delay = max(delays.values()) if delays else 1.0
        norm_delay = {cid: delays[cid] / max_delay for cid in chosen}

        round_end_time = time.time()
        aggregation_start_time=time.time()

        # 2.2 Adaptive clustering by update directions (recluster per schedule)
        if (r % recluster_every == 0) and len(chosen) > 1:
            X = np.vstack(flat_deltas)  # [m, D]
            # L2-normalize for direction-based clustering
            denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            Xn = X / denom
            labels, _ = _kmeans_numpy(Xn, min(k_clusters, len(chosen)), iters=10, seed=r)
        else:
            labels = np.zeros(len(chosen), dtype=np.int64)

        # Map label per client id
        cid_to_label = {cid: int(labels[i]) for i, cid in enumerate(chosen)}

        # 2.3 Intra-cluster aggregation (weighted by staleness gamma)
        # Build per-cluster stacks of parameter deltas
        num_clusters = int(labels.max()) + 1 if len(chosen) > 0 else 0
        cluster_updates: List[Dict[str, torch.Tensor]] = []

        for k_idx in range(num_clusters):
            members = [cid for cid in chosen if cid_to_label[cid] == k_idx]
            if not members:
                continue
            # Weighted mean of deltas in the cluster
            gamma_list = []
            stack: Dict[str, List[torch.Tensor]] = {name: [] for name in global_sd_cpu.keys()}
            for cid in members:
                gamma = math.exp(-staleness_alpha * norm_delay[cid])
                gamma_list.append(gamma)
                for name in stack:
                    stack[name].append(deltas[cid][name] * gamma)
            # compute weighted mean: sum / sum_gamma
            sum_gamma = max(1e-8, float(np.sum(gamma_list)))
            upd = {}
            for name in stack:
                s = torch.stack(stack[name], dim=0).sum(dim=0)
                upd[name] = s / sum_gamma
            cluster_updates.append(upd)

        if not cluster_updates:
            logging.warning("No cluster updates formed; skipping aggregation.")
            continue

        # 2.4 Inter-cluster robust aggregation (trimmed mean)
        # Stack cluster updates per-parameter and apply trimmed mean
        agg_update = _zeros_like(global_sd_cpu)
        for name in global_sd_cpu.keys():
            stacked = torch.stack([u[name] for u in cluster_updates], dim=0)
            agg_update[name] = _trimmed_mean(stacked, trim_ratio=trim_ratio)

        # 2.5 Momentum smoothing on the global trajectory
        # v <- beta v + agg_update; w <- w + lr * v
        for name in v:
            v[name] = momentum_beta * v[name] + agg_update[name].to(v[name].device)
        new_global = _copy_state_dict(global_model.state_dict())
        for name in new_global:
            new_global[name] = new_global[name] + momentum_lr * v[name].to(new_global[name].device)
        global_model.load_state_dict(new_global)

        # 2.6 Straggler detection & adaptive sampling weight update
        for cid in chosen:
            d = norm_delay[cid]
            if d > straggler_theta:
                # reduce future selection probability
                participation_weight[cid] = max(0.05, participation_weight[cid] * 0.8)
            else:
                # slowly restore weight for good participants
                participation_weight[cid] = min(1.0, participation_weight[cid] * 1.05)

        # Broadcast updated global model
        for cid in range(num_clients):
            clients[cid].load_state_dict(global_model.state_dict())


        # 3) Evaluate & log
        global_model_size = calculate_model_size(global_model)
        print("Global Model Size: {:.2f} MB".format(global_model_size))

        test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(global_model, test_loader, device)
        logging.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Round {r + 1} - "
            f"Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}"
        )

        # Calculate Round Duration

        aggregation_end_time= time.time()
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
        staleness_impact = np.mean([math.exp(-staleness_alpha * delays[cid]) for cid in chosen])
        metrics_dict["Staleness Impact"].append(staleness_impact)

        # Effective Contribution Weight: sum of weighted updates
        effective_contribution_weight = np.sum([math.exp(-staleness_alpha * delays[cid]) for cid in chosen])
        metrics_dict["Effective Contribution Weight"].append(effective_contribution_weight)

        # Client Participation Rate: ratio of clients participating
        client_participation_rate = len(chosen) / num_clients
        metrics_dict["Client Participation Rate"].append(client_participation_rate)

        # Client Update Time: average time spent on client updates
        client_update_time = total_update_time / len(chosen)
        metrics_dict["Client Update Time"].append(client_update_time)

        # Cumulative Delay: sum of all client delays
        metrics_dict["Cumulative Delay"].append(cumulative_delay)


        cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv = monitor_resources()
        logging_round_statics(
            r,
            global_model_size,
            test_accuracy,
            test_f1,
            test_precision,
            test_recall,
            cpu_usage,
            ram_usage,
            gpu_allocate,
            gpu_usage_percent,
            bytes_sent,
            bytes_recv,
            r_duration=r_duration,
            client_delay=0,
            throughput=throughput,
            comm_overhead=comm_overhead
        )

        # Confusion matrix
        predictions, true_labels = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.squeeze().cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        save_confusion_matrix(true_labels, predictions, r + 1, log_file)

        # Store metrics
        metrics_dict["Round"].append(r + 1)
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

    # Save the final model after all rounds
    model_path = os.path.join(log_file, "global_model_dfl_clac_star.pth")
    torch.save(global_model.state_dict(), model_path)


if __name__ == "__main__":
    fashionmnist_dfl_clac_star(
        alpha=0.5,
        num_classes=10,
        num_clients=50,
        num_rounds=5,
        num_epochs=1,
        batch_size=128,
        k_clusters=6,
        selected_ratio=0.9,
        trim_ratio=0.1,
        staleness_alpha=0.25,
        momentum_beta=0.9,
        momentum_lr=1.0,
        recluster_every=1,
        straggler_theta=0.75,
        topology="FashionMNIST_DFL_CLAC_Star",
    )
