import copy
import numpy as np
import torch
from utils import get_global_proxy
from model import MLP, GCN, SGC, SAGE, Encoder, Classifier


class Server(object):
    def __init__(self, client_list, client_ids, trainIdx, dataset, hidden, device, proxy_dim):
        self.client_list = client_list
        self.client_ids = client_ids
        self.trainIdx = trainIdx
        self.dataset = dataset
        self.classes = [c for c in range(dataset.num_classes)]
        self.num_train_nodes = [len(self.trainIdx[client_id]) for client_id in self.client_ids]
        self.coefficients = [num_train_nodes / sum(self.num_train_nodes) for num_train_nodes in self.num_train_nodes]
        self.encoder = Encoder(in_channel=dataset.num_node_features, out_channel=proxy_dim).to(device)
        self.classifier = Classifier(in_channel=proxy_dim, out_channel=dataset.num_classes).to(device)
        self.classifier2 = Classifier(in_channel=proxy_dim, out_channel=dataset.num_classes).to(device)
        self.train_label_count = [client.label_dist for client in self.client_list]
        self.proxy_dim = proxy_dim
        self.device = device

    def train(self, rounds):
        best_val_loss = 1e6
        global_proxy = [0.01 * c * torch.ones(self.proxy_dim).to(self.device) for c in self.classes]
        for round in range(1, rounds+1):
            encoder_averaged_weights = {}
            classifier_averaged_weights = {}
            classifier2_averaged_weights = {}
            local_proxy = dict()
            for i, client in enumerate(self.client_list):
                # collect updated parameters from client i
                encoder_weight, classifier_weight, classifier2_weight, proxy = client.local_update(copy.deepcopy(self.encoder), copy.deepcopy(self.classifier), copy.deepcopy(self.classifier2), global_proxy, round)

                # average parameters
                for key in self.encoder.state_dict().keys():
                    if key in encoder_averaged_weights.keys():
                        encoder_averaged_weights[key] += self.coefficients[i] * encoder_weight[key]
                    else:
                        encoder_averaged_weights[key] = self.coefficients[i] * encoder_weight[key]

                for key in self.classifier.state_dict().keys():
                    if key in classifier_averaged_weights.keys():
                        classifier_averaged_weights[key] += self.coefficients[i] * classifier_weight[key]
                    else:
                        classifier_averaged_weights[key] = self.coefficients[i] * classifier_weight[key]

                for key in self.classifier2.state_dict().keys():
                    if key in classifier2_averaged_weights.keys():
                        classifier2_averaged_weights[key] += self.coefficients[i] * classifier2_weight[key]
                    else:
                        classifier2_averaged_weights[key] = self.coefficients[i] * classifier2_weight[key]

                local_proxy[i] = proxy

            global_proxy = get_global_proxy(local_proxy, self.classes, self.client_ids, self.train_label_count)

            self.encoder.load_state_dict(encoder_averaged_weights)
            self.classifier.load_state_dict(classifier_averaged_weights)
            self.classifier2.load_state_dict(classifier2_averaged_weights)

            loss_list = []
            val_loss_list = []
            num_val_list = []
            num_test_list = []
            correct_train_list = []
            correct_val_list = []
            correct_test_list = []
            for i, client in enumerate(self.client_list):
                loss, val_loss, num_val, num_test, correct_train, correct_val, correct_test = client.stats()
                loss_list.append(loss)
                val_loss_list.append(val_loss)
                num_val_list.append(num_val)
                num_test_list.append(num_test)
                correct_train_list.append(correct_train)
                correct_val_list.append(correct_val)
                correct_test_list.append(correct_test)

            total_val = np.sum(num_val_list)
            total_test = np.sum(num_test_list)
            train_loss = np.sum(loss_list) / np.sum(self.num_train_nodes)
            val_loss = np.sum(val_loss_list) / total_val
            acc_train = np.sum(correct_train_list) / np.sum(self.num_train_nodes)
            acc_val = np.sum(correct_val_list) / total_val
            acc_test = np.sum(correct_test_list) / total_test

            print('Round: {:4d} | train_loss: {:9.5f} | val_loss: {:9.5f} | acc_train: {:7.5f} | acc_val: {:7.5f} | acc_test: {:7.5f}'
                                .format(round, train_loss, val_loss, acc_train, acc_val, acc_test))

            if val_loss < best_val_loss - 0:
                best_val_loss = val_loss
                best_test_acc = acc_test

                num_minority_list = []
                correct_minority_list = []
                for i, client in enumerate(self.client_list):
                    correct_minority, num_minority = client.print_count_nodes_per_class()
                    num_minority_list.append(num_minority)
                    correct_minority_list.append(correct_minority)
                acc_minority = np.sum(correct_minority_list) / np.sum(num_minority_list)

        return best_test_acc, acc_minority