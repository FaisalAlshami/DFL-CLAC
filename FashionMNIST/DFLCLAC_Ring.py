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

#################
# A tiny message container to simulate neighbor exchange
class _Msg:
    def __init__(self, sender: int, round_sent: int, delta: Dict[str, torch.Tensor]):
        self.sender = sender
        self.round_sent = round_sent
        self.delta = delta

def _state_dict_mean(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    assert len(state_dicts) > 0
    names = state_dicts[0].keys()
    out = {k: torch.zeros_like(state_dicts[0][k]) for k in names}
    for sd in state_dicts:
        for k in names:
            out[k] += sd[k]
    for k in out:
        out[k] /= float(len(state_dicts))
    return out

def _sd_add_(dst: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor], scale: float = 1.0):
    for k in dst.keys():
        dst[k] += scale * src[k].to(dst[k].device)

def _sd_mul_(sd: Dict[str, torch.Tensor], scale: float):
    for k in sd.keys():
        sd[k] *= scale

def _zeros_like_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in sd.items()}


def fashionmnist_dfl_clac_ring(
    alpha: float = 0.5,
    num_classes: int = 10,
    num_clients: int = 20,
    num_rounds: int = 20,
    num_epochs: int = 1,
    batch_size: int = 32,
    k_clusters: int = 5,
    selected_ratio: float = 1.0,   # ring assumes all (or most) clients act each round

    # Staleness & momentum
    staleness_alpha: float = 0.05,  # gamma = exp(-alpha * delay)
    momentum_beta: float = 0.9,     # per-client momentum buffer
    momentum_lr: float = 1.0,

    # Reclustering overlay & stragglers
    recluster_every: int = 5,       # optional overlay (cluster-consensus) frequency
    straggler_theta: float = 0.75,  # normalized-delay threshold
    link_drop_prob: float = 0.05,   # per-message drop prob
    delay_low: float = 0.0,         # seconds, simulated extra delay
    delay_high: float = 0.1,

    topology: str = "Ring",
    dirName: str = "FashionMNIST_DFL_CLAC_Ring",
):
    """
    Simulate DFL-CLAC on MNIST_DFL_CLAC_Ring under a *Ring* (decentralized) topology.
    Key differences from Star:
      • No central server/global_model; every client maintains its own model.
      • Each round: client trains locally, sends *delta* to two neighbors.
      • Each client aggregates: own delta + neighbor deltas, weighted by staleness gamma.
      • Momentum per client smooths the update trajectory.
      • Optional overlay: every `recluster_every` rounds, do a light cluster-consensus step
        (k-means on update directions; within each cluster, nudge models toward the cluster mean).
      • For evaluation, construct a *pseudo-global* by averaging client parameters.
    """

    # ---------------------------
    # 0) Setup
    # ---------------------------
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_file = setup_logging(dirName, topology)
    logging.info(f"Starting DFL-CLAC (Ring). Log dir: {log_file}")

    # ---------------------------
    # 1) Data: MNIST loaders
    # ---------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std for grayscale
    ])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client_loaders, valid_client_indices = non_iid_distribution(num_clients, num_classes, train_data, batch_size)

    # ---------------------------
    # 2) Initialize client models
    # ---------------------------
    # All clients start from the same initialization
    base_model = MNISTModelCNN().to(device)
    base_sd = base_model.state_dict()

    clients: List[nn.Module] = [MNISTModelCNN().to(device) for _ in range(num_clients)]
    for c in clients:
        c.load_state_dict(base_sd)

    # Momentum buffers per client
    v_list: List[Dict[str, torch.Tensor]] = [_zeros_like(base_sd) for _ in range(num_clients)]

    # For staleness & sampling
    participation_weight = np.ones(num_clients, dtype=np.float64)

    # ---------------------------
    # 3) Metrics storage
    # ---------------------------
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
        "Round Duration (sec)": [],
        "Aggregation Duration (sec)": [],
        "Communication Overhead (bytes)": [],
        "Client Delays (sec)": [],
        "Throughput": [],
        "Straggler Rate": [],
        "Staleness Impact": [],
        "Effective Contribution Weight": [],
        "Client Participation Rate": [],
        "Client Update Time": [],
        "Cumulative Delay": [],
    }

    # Precompute ring neighbors
    def neighbors(i: int) -> Tuple[int, int]:
        return ( (i - 1) % num_clients, (i + 1) % num_clients )

    # ---------------------------
    # 4) Federated rounds
    # ---------------------------
    for r in range(num_rounds):
        round_start_time = time.time()
        logging.info(f"===== Round {r+1}/{num_rounds} =====")
        print(f"Round {r+1}/{num_rounds}")

        # Select participating clients
        m = max(1, int(selected_ratio * len(valid_client_indices)))
        weights = participation_weight[valid_client_indices]
        probs = weights / (weights.sum() + 1e-12)
        chosen = np.random.choice(valid_client_indices, size=m, replace=False, p=probs)

        # For each chosen client: local training → compute delta wrt *its own* pre-round weights
        client_pre_sds = [ _copy_state_dict(clients[i].state_dict()) for i in range(num_clients) ]

        deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        flat_deltas: List[np.ndarray] = []
        delays_sec: Dict[int, float] = {}

        total_update_time, total_delay_time, straggler_count, cumulative_delay = 0.0, 0.0, 0, 0.0

        # Local training
        for cid in chosen:
            model = clients[cid]
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            client_start = time.time()

            # train on that client's partition
            loader = client_loaders[valid_client_indices.index(cid)]
            for _ in range(num_epochs):
                _loss, _acc = train_model(model, loader, nn.CrossEntropyLoss(), optimizer, device)

            client_finish = time.time()

            # simulate network delay for this client
            net_delay = random.uniform(delay_low, delay_high)
            #time.sleep(net_delay)
            finish_time = client_finish + net_delay

            # compute delta wrt pre-round state
            pre_sd = client_pre_sds[cid]
            new_sd = _copy_state_dict(model.state_dict())
            delta_sd = _model_delta(pre_sd, new_sd)

            deltas[cid] = delta_sd
            flat_deltas.append(_flatten_params(delta_sd).numpy())
            delays_sec[cid] = max(1e-6, finish_time - round_start_time)

            total_update_time += (client_finish - client_start)
            total_delay_time += net_delay
            cumulative_delay += net_delay
            if net_delay > 0.3:
                straggler_count += 1

        # Normalize delays for weighting decisions/logging
        max_delay = max(delays_sec.values()) if delays_sec else 1.0
        norm_delay = {cid: delays_sec[cid] / max_delay for cid in chosen}

        # ---------------------------
        # 4.1 Neighbor message passing (ring)
        # ---------------------------
        # Build outgoing messages from each chosen client to its two neighbors
        out_msgs: Dict[Tuple[int, int], _Msg] = {}
        for cid in chosen:
            l, r_neighbor = neighbors(cid)
            msg = _Msg(sender=cid, round_sent=r, delta=deltas[cid])
            out_msgs[(cid, l)] = msg
            out_msgs[(cid, r_neighbor)] = msg

        # Deliver messages with drops
        inbox = {i: [] for i in range(num_clients)}  # inbox[i] = list of _Msg received this round
        for (sender, dst), msg in out_msgs.items():
            if random.random() < link_drop_prob:
                continue
            # deliver (we could store arbitrary delay metadata; staleness uses delays_sec)
            inbox[dst].append(msg)

        # ---------------------------
        # 4.2 Each client aggregates and updates (decentralized)
        # ---------------------------
        round_end_time = time.time()
        aggregation_start = time.time()

        for i in range(num_clients):
            # if not chosen this round, no new local delta; use zero delta
            own_delta = deltas.get(i, _zeros_like_sd(client_pre_sds[i]))
            # start from own_delta with weight 1.0
            agg = _copy_state_dict(own_delta)
            total_w = 1.0

            # incorporate neighbor deltas with time-sensitive weights
            for msg in inbox[i]:
                # staleness factor based on that sender's normalized delay
                sender = msg.sender
                d_norm = norm_delay.get(sender, 0.0)  # 0 if not in chosen (fast/none)
                gamma = math.exp(-staleness_alpha * d_norm)
                _sd_add_(agg, msg.delta, scale=gamma)
                total_w += gamma

            # normalize
            _sd_mul_(agg, 1.0 / max(total_w, 1e-8))

            # momentum update on *client i*
            v_i = v_list[i]
            for k in v_i.keys():
                v_i[k] = momentum_beta * v_i[k] + agg[k].to(v_i[k].device)
            new_sd = _copy_state_dict(clients[i].state_dict())
            for k in new_sd.keys():
                new_sd[k] = new_sd[k] + momentum_lr * v_i[k].to(new_sd[k].device)
            clients[i].load_state_dict(new_sd)

        # ---------------------------
        # 4.3 Optional overlay: cluster-consensus step for CLAC
        # ---------------------------
        if recluster_every > 0 and (r % recluster_every == 0) and len(chosen) > 1:
            X = np.vstack(flat_deltas) if flat_deltas else None
            if X is not None:
                denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                Xn = X / denom
                labels, _ = _kmeans_numpy(Xn, min(k_clusters, len(chosen)), iters=10, seed=r)
                cid_to_label = {cid: int(labels[i]) for i, cid in enumerate(chosen)}

                # For each cluster: nudge members toward the cluster-average *model*
                for cl in range(int(labels.max()) + 1):
                    members = [cid for cid in chosen if cid_to_label[cid] == cl]
                    if len(members) <= 1:
                        continue
                    sds = [clients[cid].state_dict() for cid in members]
                    mean_sd = _state_dict_mean(sds)
                    # move 20% toward the mean
                    for cid in members:
                        sd = _copy_state_dict(clients[cid].state_dict())
                        for k in sd.keys():
                            sd[k] = 0.8 * sd[k] + 0.2 * mean_sd[k].to(sd[k].device)
                        clients[cid].load_state_dict(sd)

        aggregation_end = time.time()

        # ---------------------------
        # 5) Evaluation (pseudo-global = average of client params)
        # ---------------------------
        avg_sd = _state_dict_mean([clients[i].state_dict() for i in range(num_clients)])
        pseudo_global = MNISTModelCNN().to(device)
        pseudo_global.load_state_dict(avg_sd)

        global_model_size = calculate_model_size(pseudo_global)
        print("Pseudo-Global Model Size: {:.2f} MB".format(global_model_size))

        test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(pseudo_global, test_loader, device)
        logging.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Round {r + 1} - "
            f"Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}"
        )

        # ---------------------------
        # 6) Metrics & logging
        # ---------------------------
        #round_end_time = time.time()
        r_duration = round_end_time - round_start_time
        aggregation_duration = aggregation_end - aggregation_start

        metrics_dict["Round"].append(r + 1)
        metrics_dict["Accuracy"].append(test_accuracy)
        metrics_dict["Precision"].append(test_precision)
        metrics_dict["Recall"].append(test_recall)
        metrics_dict["F1 Score"].append(test_f1)
        metrics_dict["Round Duration (sec)"].append(r_duration)
        metrics_dict["Aggregation Duration (sec)"].append(aggregation_duration)

        # Average client delay this round
        client_delay = np.mean(list(delays_sec.values())) if delays_sec else 0.0
        metrics_dict["Client Delays (sec)"].append(client_delay)

        # Simple comm overhead proxy: number of msgs * params-per-delta (approx with param-count)
        # Here we just use number of delivered messages.
        delivered = sum(len(inbox[i]) for i in range(num_clients))
        metrics_dict["Communication Overhead (bytes)"].append(delivered)

        throughput = len(chosen) / (r_duration / 60 if r_duration > 0 else 1.0)
        metrics_dict["Throughput"].append(throughput)

        straggler_rate = straggler_count / max(1, len(chosen))
        metrics_dict["Straggler Rate"].append(straggler_rate)

        staleness_impact = np.mean([math.exp(-staleness_alpha * delays_sec[cid]) for cid in chosen]) if chosen.size else 1.0
        metrics_dict["Staleness Impact"].append(staleness_impact)
        effective_contribution_weight = np.sum([math.exp(-staleness_alpha * delays_sec[cid]) for cid in chosen]) if chosen.size else 0.0
        metrics_dict["Effective Contribution Weight"].append(effective_contribution_weight)

        client_participation_rate = len(chosen) / num_clients
        metrics_dict["Client Participation Rate"].append(client_participation_rate)
        client_update_time = total_update_time / max(1, len(chosen))
        metrics_dict["Client Update Time"].append(client_update_time)
        metrics_dict["Cumulative Delay"].append(cumulative_delay)

        cpu_usage, ram_usage, gpu_allocate, gpu_usage_percent, bytes_sent, bytes_recv = monitor_resources()
        metrics_dict["CPU Usage"].append(cpu_usage)
        metrics_dict["RAM Usage"].append(ram_usage)
        metrics_dict["GPU Allocated"].append(gpu_allocate)
        metrics_dict["GPU Usage"].append(gpu_usage_percent)

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
            client_delay=client_delay,
            throughput=throughput,
            comm_overhead=delivered
        )

        # Confusion matrix on pseudo-global
        predictions, true_labels = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), data.new_tensor(target, dtype=torch.long).to(device)
                output = pseudo_global(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.squeeze().cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        save_confusion_matrix(true_labels, predictions, r + 1, log_file)

        visualization_save_metrics(metrics_dict, log_file)

        # 6.1 Adaptive participation tweak (straggler handling, decentralized)
        for cid in chosen:
            d = norm_delay.get(cid, 0.0)
            if d > straggler_theta:
                participation_weight[cid] = max(0.05, participation_weight[cid] * 0.8)
            else:
                participation_weight[cid] = min(1.0, participation_weight[cid] * 1.05)

    # Save the final pseudo-global model
    model_path = os.path.join(log_file, "global_model_dfl_clac_ring.pth")
    torch.save(pseudo_global.state_dict(), model_path)
    return metrics_dict, model_path

if __name__ == "__main__":
    fashionmnist_dfl_clac_ring(
        alpha=0.5,
        num_classes=100,
        num_clients=50,
        num_rounds=5,
        num_epochs=1,
        batch_size=128,
        k_clusters=6,
        selected_ratio=0.9,
        #trim_ratio=0.1,
        staleness_alpha=0.25,
        momentum_beta=0.9,
        momentum_lr=1.0,
        recluster_every=1,
        straggler_theta=0.75,
        topology="Ring",
        dirName="FashionMNIST_DFL_CLAC",
    )
