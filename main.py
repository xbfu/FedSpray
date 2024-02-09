import os
import numpy as np
import torch

from data import load_data
from server import Server
from client import Client


def run(dataset_name, epochs, hidden, lr, lambda_1, proxy_dim, seed):
    arch_name = os.path.basename(__file__).split('.')[0]
    data_path = './data/'
    file_names = sorted(os.listdir('./partition/'))

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset, num_clients, trainIdx, valIdx, testIdx = load_data(dataset_name, data_path, file_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_ids = [i for i in range(num_clients)]
    client_list = [Client(client, dataset, trainIdx[client], valIdx[client], testIdx[client], lr, epochs, hidden, lambda_1, proxy_dim, device) for client in client_ids]

    server = Server(client_list, client_ids, trainIdx, dataset, hidden, device, proxy_dim)
    best_test_acc, acc_minority = server.train(rounds=500)
    print('Arch: {:s} | dataset: {:s} | lr: {:6.4f} | epochs:{:2d} | lambda_1:{:5.1f} | proxy_dim: {:3d} | seed: {:2d} | best_test_acc: {:6.4f} | acc_minority: {:6.4f}'
          .format(arch_name, dataset_name, lr, epochs, lambda_1, proxy_dim, seed, best_test_acc, acc_minority))


if __name__ == '__main__':
    dataset_name = 'PubMed'
    epochs = 5
    lambda_1 = 5
    run(dataset_name, epochs, hidden=64, lr=0.003, lambda_1=1.0, proxy_dim=64, seed=0)
