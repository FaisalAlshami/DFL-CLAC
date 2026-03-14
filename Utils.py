from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from openpyxl import Workbook
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
import psutil
import GPUtil

# Set up logging and create a unique folder for each run
def setup_logging(dirName, topologyName):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs\clustering_experiements", dirName + "_" + topologyName + "_" + current_time)  # Create timestamped subfolder inside logs/
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

    log_file = os.path.join(log_dir, dirName+"_"+topologyName+"_"+current_time+".log")  # Log file inside the new folder
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    print("Save in "+log_dir)
    return log_dir  # Return the folder path for saving models and logs



def visualization_save_metrics(metrics_dict, dir_path):
    # Save metrics to an Excel file
    #metrics_df = pd.DataFrame(metrics_dict)
    #metrics_df.to_excel(f"{dir_path}/metrics.xlsx", index=False)
    # Create a new workbook and select the active worksheet
    wb = Workbook()
    ws = wb.active

    # Write the header row
    ws.append(list(metrics_dict.keys()))

    # Write the data rows
    for i in range(len(metrics_dict['Round'])):
        row = [metrics_dict[key][i] for key in metrics_dict.keys()]
        ws.append(row)

    # Save the workbook
    wb.save(f"{dir_path}/metrics.xlsx")


    ############
    # save metrics to images with high-quality graphs
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CPU Usage', 'RAM Usage', 'GPU Allocated',
                   'GPU Usage']:
        plt.figure(figsize=(10, 6), dpi=300)  # High-quality figure
        plt.plot(metrics_dict['Round'], metrics_dict[metric], marker='o', linestyle='-', linewidth=2, label=metric)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} over Rounds', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save the figure as an individual image
        plt.savefig(f"{dir_path}/{metric.lower().replace(' ', '_')}.png", dpi=200, bbox_inches='tight')
        plt.close()

def visualization_save_metrics_dacey(metrics_dict, dir_path):
    # Save metrics to an Excel file
    #metrics_df = pd.DataFrame(metrics_dict)
    #metrics_df.to_excel(f"{dir_path}/metrics.xlsx", index=False)
    # Create a new workbook and select the active worksheet
    wb = Workbook()
    ws = wb.active

    # Write the header row
    ws.append(list(metrics_dict.keys()))

    # Write the data rows
    for i in range(len(metrics_dict['Round'])):
        row = [metrics_dict[key][i] for key in metrics_dict.keys()]
        ws.append(row)

    # Save the workbook
    wb.save(f"{dir_path}/metrics.xlsx")


    ############
    # save metrics to images with high-quality graphs
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CPU Usage', 'RAM Usage', 'GPU Allocated',
                   'GPU Usage', 'Round Duration (sec)', 'Communication Overhead (bytes)', 'Throughput']:
        plt.figure(figsize=(10, 6), dpi=300)  # High-quality figure
        plt.plot(metrics_dict['Round'], metrics_dict[metric], marker='o', linestyle='-', linewidth=2, label=metric)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} over Rounds', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save the figure as an individual image
        plt.savefig(f"{dir_path}/{metric.lower().replace(' ', '_')}.png", dpi=200, bbox_inches='tight')
        plt.close()


def save_confusion_matrix(y_true, y_pred, round_num, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Round {round_num}')
    cm_filename = os.path.join(save_dir, f"confusion_matrix_round_{round_num}.png")
    plt.savefig(cm_filename, dpi=200)
    plt.close()

def save_confusion_matrix_cifar100(
    y_true, y_pred, round_num, save_dir,
    class_names=None, normalize=True, max_classes_displayed=20
):
    """
    Saves a confusion matrix heatmap (without annotations) and exports the matrix to CSV.

    Parameters:
    - y_true: Ground truth labels (list or array)
    - y_pred: Predicted labels (list or array)
    - round_num: Round or epoch number for file naming
    - save_dir: Directory to save the PNG and CSV files
    - class_names: Optional list of 100 class names (CIFAR-100)
    - normalize: Whether to normalize each row (per-class accuracy view)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row if requested
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # If class names not provided, use indices
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    # Save CSV file
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(save_dir, f'confusion_matrix_round_{round_num}.csv')
    cm_df.to_csv(csv_path)

    # Save heatmap without cell annotations for better readability
    plt.figure(figsize=(30, 24))  # large figure for CIFAR-100
    sns.heatmap(
        cm,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot=False,
        cbar=True
    )
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title(f'Confusion Matrix - Round {round_num}', fontsize=20)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    # Save PNG
    fig_path = os.path.join(save_dir, f'confusion_matrix_round_{round_num}.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
# Try importing pynvml for GPU tracking
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def monitor_resources():
    process = psutil.Process(os.getpid())

    # 🔹 1️⃣ **Fix: Reliable CPU Usage for Current Process**
    cpu_times1 = process.cpu_times()
    total_cpu_time1 = cpu_times1.user + cpu_times1.system  # User + System time
    start_time = time.time()  # Record the start time

    time.sleep(1)  # Allow time to capture CPU usage

    cpu_times2 = process.cpu_times()
    total_cpu_time2 = cpu_times2.user + cpu_times2.system  # User + System time
    elapsed_time = time.time() - start_time  # Time difference

    # Compute CPU usage over time
    cpu_usage = ((total_cpu_time2 - total_cpu_time1) / elapsed_time) * 100
    cpu_usage = cpu_usage / psutil.cpu_count(logical=True)  # Normalize across cores

    # 🔹 2️⃣ **Fix: RAM Usage for Current Process in MB**
    ram_usage_mb = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB

    # 🔹 3️⃣ **Fix: GPU Usage for Current Process**
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert to GB

        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                pid = os.getpid()  # Current process ID

                # Get per-process GPU utilization
                gpu_util_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                gpu_usage_percentage = sum([p.gpuUtil for p in gpu_util_list if p.pid == pid])

                # Fallback if 0
                if gpu_usage_percentage == 0:
                    gpu_usage_percentage = (gpu_allocated / (torch.cuda.max_memory_allocated() / 1024 ** 3)) * 100
            except:
                gpu_usage_percentage = "N/A (Restricted Access)"
        else:
            gpu_usage_percentage = "N/A (pynvml not installed)"
    else:
        gpu_allocated = 0
        gpu_usage_percentage = 0

    # 🔹 4️⃣ **Network Usage (Bytes Sent & Received in MB)**
    net1 = psutil.net_io_counters()
    time.sleep(1)
    net2 = psutil.net_io_counters()

    bytes_sent = (net2.bytes_sent - net1.bytes_sent) / (1024 ** 2)  # Convert bytes to MB
    bytes_recv = (net2.bytes_recv - net1.bytes_recv) / (1024 ** 2)  # Convert bytes to MB

    return cpu_usage, ram_usage_mb, gpu_allocated, gpu_usage_percentage, bytes_sent, bytes_recv


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

    # Print results for debugging
    # logging.info(f"Round {round_num + 1}: r_duration = {r_duration:.4f}, "
    #              f"client_delay = {client_delay:.4f}, "
    #              f"throughput = {throughput:.4f}, "
    #              f"comm_overhead = {comm_overhead:.4f}")
    #
    #
    # print(f"Round {round_num + 1}: r_duration = {r_duration:.4f}, "
    #              f"client_delay = {client_delay:.4f}, "
    #              f"throughput = {throughput:.4f}, "
    #              f"comm_overhead = {comm_overhead:.4f}")

    logging.info("-" * 50 + "\n")



