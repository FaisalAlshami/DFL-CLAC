import torch

from CIFAR10.DFLCLAC_Ring import cifar10_dfl_clac_ring
from CIFAR10.DFLCLAC_Star import cifar10_dfl_clac_star
from CIFAR10.FedAvg_Ring import cifar10_fedavg_ring
from CIFAR10.FedAvg_Star import cifar10_fedavg_star

from CIFAR100.DFLCLAC_Ring import cifar100_dfl_clac_ring
from CIFAR100.DFLCLAC_Star import cifar100_dfl_clac_star
from CIFAR100.FedAvg_Ring import cifar100_fedavg_ring
from CIFAR100.FedAvg_Star import cifar100_fedavg_star


from FashionMNIST.DFLCLAC_Ring import fashionmnist_dfl_clac_ring
from FashionMNIST.DFLCLAC_Star import fashionmnist_dfl_clac_star
from FashionMNIST.FedAvg_Ring import fashionmnist_fedavg_ring
from FashionMNIST.FedAvg_Star import fashionmnist_fedavg_star

from MNIST.DFLCLAC_Ring import mnist_dfl_clac_ring
from MNIST.DFLCLAC_Star import mnist_dfl_clac_star
from MNIST.FedAvg_Ring import mnist_fedavg_ring
from MNIST.FedAvg_Star import mnist_fedavg_star

if __name__ == "__main__":
    print(torch.__version__)  # Check the version
    print(torch.cuda.is_available())  # Should return True if CUDA is working

    #efficient_experiemnts()
    #test_monitor_resources()
    num_rounds=100
    k=1
    nc=[50, 30, 40, 60]
    for n in nc:
        cifar10_fedavg_star(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                            batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                            aggregation="FedAvg", topology="Star", dirName=f"CIFAR10_FedAvg_{n}_clients")

        cifar10_fedavg_ring(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                            batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                            aggregation="FedAvg", topology="Ring", dirName=f"CIFAR10_FedAvg_{n}_Ring")
        cifar10_dfl_clac_star(
            alpha=0.5,
            num_classes=10,
            num_clients=n,
            num_rounds=num_rounds,
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
            topology="Star",
            dirName=f"CIFAR10_DFL_CLAC_{n}_Star",
        )

        cifar10_dfl_clac_ring(
            alpha=0.5,
            num_classes=10,
            num_clients=n,
            num_rounds=num_rounds,
            num_epochs=1,
            batch_size=128,
            k_clusters=6,
            selected_ratio=0.9,
            # trim_ratio=0.1,
            staleness_alpha=0.25,
            momentum_beta=0.9,
            momentum_lr=1.0,
            recluster_every=1,
            straggler_theta=0.75,
            topology="Star",
            dirName=f"CIFAR10_DFL_CLAC_{n}_ring",
        )
        #################
        cifar100_fedavg_ring(alpha=0.5, num_classes=100, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                             batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                             aggregation="FedAvg", topology="Ring", dirName=F"CIFAR100_FedAvg_{n}_Ring")

        cifar100_fedavg_star(alpha=0.5, num_classes=100, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                             batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                             aggregation="FedAvg", topology="Star", dirName=f"CIFAR100_FedAvg_{n}_Star")
        cifar100_dfl_clac_star(
            alpha=0.5,
            num_classes=100,
            num_clients=n,
            num_rounds=num_rounds,
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
            topology="Star",
            dirName=f"CIFAR100_DFL_CLAC_{n}_Star",
        )
        cifar100_dfl_clac_ring(
            alpha=0.5,
            num_classes=100,
            num_clients=n,
            num_rounds=num_rounds,
            num_epochs=1,
            batch_size=128,
            k_clusters=6,
            selected_ratio=0.9,
            # trim_ratio=0.1,
            staleness_alpha=0.25,
            momentum_beta=0.9,
            momentum_lr=1.0,
            recluster_every=1,
            straggler_theta=0.75,
            topology="Ring",
            dirName=f"CIFAR100_DFL_CLAC_{n}_Ring",
        )
        ###############
        mnist_fedavg_star(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                          batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                          aggregation="FedAvg", topology="Star", dirName=f"MNIST_FedAvg_{n}_Star")

        mnist_fedavg_ring(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                          batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                          aggregation="FedAvg", topology="Ring", dirName=f"MNIST_FedAvg_{n}_Ring")

        mnist_dfl_clac_star( alpha=0.5,num_classes=10,num_clients=n,num_rounds=num_rounds,
                             num_epochs=1,batch_size=128,k_clusters=6,selected_ratio=0.9,trim_ratio=0.1,
                             staleness_alpha=0.25,momentum_beta=0.9, momentum_lr=1.0, recluster_every=1,
                             straggler_theta=0.75, topology=f"MNIST_DFL_CLAC_{n}_Star",
        )
        mnist_dfl_clac_ring( alpha=0.5, num_classes=100, num_clients=n, num_rounds=num_rounds, num_epochs=1,
                             batch_size=128, k_clusters=6, selected_ratio=0.9,  # trim_ratio=0.1,
                             staleness_alpha=0.25, momentum_beta=0.9, momentum_lr=1.0, recluster_every=1,
                             straggler_theta=0.75, topology="Ring", dirName=f"MNIST_DFL_CLAC_{n}_Ring",
        )

       #######
        fashionmnist_fedavg_star(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                             batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9, Allow_delay=180,
                             aggregation="FedAvg", topology="Star", dirName=f"FashionMNIST_FedAvg_{n}_Star")
        fashionmnist_fedavg_ring(alpha=0.5, num_classes=10, num_clients=n, num_rounds=num_rounds, num_epochs=20,
                                 batch_size=128, cluster_num=12, num_components=100, selected_ratio=0.9,
                                 Allow_delay=180,aggregation="FedAvg", topology="Ring",
                                 dirName=f"FashionMNIST_FedAvg_{n}_Ring")
        fashionmnist_dfl_clac_star(
            alpha=0.5,
            num_classes=10,
            num_clients=n,
            num_rounds=num_rounds,
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
            topology=f"FashionMNIST_DFL_CLAC_{n}_Star",
        )
        fashionmnist_dfl_clac_ring(
            alpha=0.5,
            num_classes=100,
            num_clients=n,
            num_rounds=num_rounds,
            num_epochs=1,
            batch_size=128,
            k_clusters=6,
            selected_ratio=0.9,
            # trim_ratio=0.1,
            staleness_alpha=0.25,
            momentum_beta=0.9,
            momentum_lr=1.0,
            recluster_every=1,
            straggler_theta=0.75,
            topology="Ring",
            dirName=f"FashionMNIST_DFL_CLAC_{n}_Ring",
        )

