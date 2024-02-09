import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import subgraph
from model import MLP, GCN, SGC, SAGE, Encoder, Classifier


class Client(object):
    def __init__(self, client_id, dataset, trainIdx, valIdx, testIdx, lr, epochs, hidden, lambda_1, proxy_dim, device):
        self.client_id = client_id
        self.node_list = trainIdx + valIdx + testIdx
        self.data = dataset[0]
        self.trainIdx = list(range(0, len(trainIdx)))
        self.valIdx = list(range(len(trainIdx), len(trainIdx) + len(valIdx)))
        self.testIdx = list(range(len(trainIdx) + len(valIdx), len(trainIdx) + len(valIdx) + len(testIdx)))
        self.features = self.data.x[self.node_list]
        self.labels = self.data.y[self.node_list]
        self.features = self.features.to(device)
        self.labels = self.labels.squeeze().to(device)
        self.classes = [c for c in range(dataset.num_classes)]
        self.lr = lr
        self.epochs = epochs
        self.lambda_1 = lambda_1
        self.device = device
        self.proxy_dim = proxy_dim
        self.label_dist = [len(torch.where(self.labels[self.trainIdx] == c)[0]) / len(self.trainIdx) for c in self.classes]
        self.majority_class = np.argmax(self.label_dist)

        self.encoder = Encoder(in_channel=dataset.num_node_features, out_channel=proxy_dim).to(device)
        self.classifier = Classifier(in_channel=proxy_dim, out_channel=dataset.num_classes).to(device)
        self.classifier2 = Classifier(in_channel=proxy_dim, out_channel=dataset.num_classes).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()) + list(self.classifier2.parameters()), lr=self.lr)

        self.gnn = SAGE(in_channel=dataset.num_node_features, out_channel=dataset.num_classes, hidden=hidden).to(device)
        self.optimizer_per = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

        self.subgraph = subgraph(subset=torch.tensor(self.node_list, dtype=torch.long), edge_index=self.data.edge_index,
                                 relabel_nodes=True,  num_nodes=self.data.num_nodes)
        self.edge_index = self.subgraph[0].to(device)

        self.proxy = torch.full((len(self.node_list), proxy_dim), 0.1, requires_grad=True, device=device)
        self.optimizer_proxy = torch.optim.Adam([self.proxy], lr=0.02)

    def local_update(self, encoder, classifier, classifier2, proxy, round):
        criterion_t = nn.KLDivLoss(reduction="batchmean", log_target=True)
        labels = self.labels[self.trainIdx]
        self.encoder.load_state_dict(encoder.state_dict())
        self.classifier.load_state_dict(classifier.state_dict())
        self.classifier2.load_state_dict(classifier.state_dict())
        emb_mlp = encoder(self.features)
        output_mlp2 = classifier2(emb_mlp)
        p, indices = F.softmax(output_mlp2, dim=1)[self.valIdx + self.testIdx].max(dim=1)
        reliableIdx = self.trainIdx + (torch.where(p > 0.7)[0] + len(self.trainIdx)).tolist()
        weighted_proxy = sum([F.softmax(output_mlp2, dim=1)[:, c][:, None] * proxy[c].data for c in self.classes])

        pseudo_label = 0 * self.labels
        pseudo_label[self.trainIdx] = self.labels[self.trainIdx]
        pseudo_label[self.valIdx + self.testIdx] = indices

        with torch.no_grad():
            for c in self.classes:
                self.proxy[self.valIdx + self.testIdx] = weighted_proxy[self.valIdx + self.testIdx]

        temp = copy.deepcopy(self.proxy)

        emb_mlp = encoder(self.features)
        output_new = classifier(emb_mlp, temp)

        self.gnn.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer_per.zero_grad()
            output_gnn, emb_gnn = self.gnn(self.features, self.edge_index)
            ce_loss = F.cross_entropy(output_gnn[self.trainIdx], self.labels[self.trainIdx])
            dist = criterion_t(F.log_softmax(output_gnn[reliableIdx], dim=1), F.log_softmax(output_new[reliableIdx], dim=1).detach())

            loss = 1 * ce_loss + self.lambda_1 * dist
            loss.backward()
            self.optimizer_per.step()

        output_gnn, emb_gnn = self.gnn(self.features, self.edge_index)

        self.encoder.train()
        self.classifier.train()
        self.classifier2.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer_proxy.zero_grad()
            self.optimizer.zero_grad()
            emb_mlp = self.encoder(self.features)
            output_mlp = self.classifier(emb_mlp, self.proxy)
            output_mlp2 = self.classifier2(emb_mlp)
            ce_loss = F.cross_entropy(output_mlp2[self.trainIdx], self.labels[self.trainIdx])
            dist = criterion_t(F.log_softmax(output_mlp[self.trainIdx]/1, dim=1), F.log_softmax(output_gnn[self.trainIdx]/1, dim=1).detach())

            loss = 1 * ce_loss + 1 * dist

            loss.backward()
            self.optimizer.step()
            self.optimizer_proxy.step()

        local_proxy = dict()
        for c in self.classes:
            if len(torch.where(labels == c)[0]) > 0:
                local_proxy[c] = self.proxy[self.trainIdx][torch.where(labels == c)[0]].mean(0)
            else:
                local_proxy[c] = torch.zeros_like(proxy[0]).to(self.device)

        return self.encoder.state_dict(), self.classifier.state_dict(), self.classifier2.state_dict(), local_proxy

    def stats(self):
        self.gnn.eval()
        output, emb = self.gnn(self.features, self.edge_index)
        loss = F.cross_entropy(output[self.trainIdx], self.labels[self.trainIdx])
        val_loss = F.cross_entropy(output[self.valIdx], self.labels[self.valIdx])
        pred = output.argmax(dim=1)
        correct_train = sum(np.array(self.labels[self.trainIdx].cpu()) == np.array(pred[self.trainIdx].cpu()))
        correct_val = sum(np.array(self.labels[self.valIdx].cpu()) == np.array(pred[self.valIdx].cpu()))
        correct_test = sum(np.array(self.labels[self.testIdx].cpu()) == np.array(pred[self.testIdx].cpu()))

        return loss.item() * len(self.trainIdx), val_loss.item() * len(self.valIdx), \
               len(self.valIdx), len(self.testIdx), correct_train, correct_val, correct_test

    def print_count_nodes_per_class(self, gnn=None):
        # self.gnn.load_state_dict(gnn.state_dict())
        self.gnn.eval()
        output, emb = self.gnn(self.features, self.edge_index)
        prediction = output.argmax(dim=1)[self.testIdx]
        labels = self.labels[self.testIdx]
        majority_class = np.argmax(self.label_dist)

        label_counter = len(torch.where(labels != majority_class)[0])
        idx = torch.where(labels != majority_class)[0]
        correct = torch.sum(labels[idx] == prediction[idx])
        acc = correct / label_counter
        return correct.item(), label_counter

