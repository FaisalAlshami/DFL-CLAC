import logging
import numpy as np
from torch.utils.data import DataLoader, Subset


def non_iid_distribution(num_clients,num_classes,train_data,batch_size):

    class_data = {i: [] for i in range(num_classes)}
    for idx, (img, label) in enumerate(train_data):
        class_data[label].append(idx)


    clients_data = {i: [] for i in range(num_clients)}

    samples_per_class_per_client = len(train_data) // num_classes // num_clients


    for class_id, indices in class_data.items():
        np.random.shuffle(indices)
        num_samples_per_client = len(indices) // num_clients
        for client_id in range(num_clients):
            start_idx = client_id * num_samples_per_client
            end_idx = (client_id + 1) * num_samples_per_client if client_id != (num_clients - 1) else len(indices)
            clients_data[client_id].extend(indices[start_idx:end_idx])

    valid_client_indices = [client_id for client_id, data in clients_data.items() if len(data) > 0]

    if len(valid_client_indices) == 0:
        raise ValueError("No valid clients with data found. Check data distribution process.")

    client_loaders = {}

    for client_id in valid_client_indices:
        client_indices = clients_data[client_id]
        client_subset = Subset(train_data, client_indices)

        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders[client_id] = client_loader

        #print(f"Client {client_id} has {len(client_indices)} samples.")
        #logging.info(f"Client {client_id} has {len(client_indices)} samples.")

    client_example_loader = client_loaders[valid_client_indices[0]]
    #for batch_idx, (data, targets) in enumerate(client_example_loader):
    #    print(f"Batch {batch_idx}, Data shape: {data.shape}, Targets: {targets}")
    #    logging.info(f"Batch {batch_idx}, Data shape: {data.shape}, Targets: {targets}")
    #    break

    return client_loaders, valid_client_indices


def non_iid_distribution_cifar100(num_clients, num_classes, train_data, batch_size):
    # Group sample indices by class label
    class_data = {i: [] for i in range(num_classes)}
    for idx, (img, label) in enumerate(train_data):
        if isinstance(label, int):
            class_data[label].append(idx)
        else:
            class_data[label.item()].append(idx)  # In case label is a Tensor

    # Initialize client data containers
    clients_data = {i: [] for i in range(num_clients)}

    # Distribute each class’s samples across clients (non-IID, but still balanced)
    for class_id, indices in class_data.items():
        np.random.shuffle(indices)
        num_samples_per_client = len(indices) // num_clients
        for client_id in range(num_clients):
            start_idx = client_id * num_samples_per_client
            end_idx = (client_id + 1) * num_samples_per_client if client_id != (num_clients - 1) else len(indices)
            clients_data[client_id].extend(indices[start_idx:end_idx])

    # Ensure only valid clients are used
    valid_client_indices = [client_id for client_id, data in clients_data.items() if len(data) > 0]

    if len(valid_client_indices) == 0:
        raise ValueError("No valid clients with data found. Check data distribution process.")

    # Create DataLoader per client
    client_loaders = {}
    for client_id in valid_client_indices:
        client_indices = clients_data[client_id]
        client_subset = Subset(train_data, client_indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders[client_id] = client_loader

    return client_loaders, valid_client_indices

