import torch
from torch import Tensor
from typing import List, Optional, Union


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def subgraph(subset: Union[Tensor, List[int]], edge_index: Tensor, edge_attr: Optional[Tensor] = None, relabel_nodes: bool = False, num_nodes: Optional[int] = None):

    device = edge_index.device
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset] = 1

    if relabel_nodes:
        node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        node_idx[subset] = torch.arange(subset.size(0), device=device)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    edge_index = node_idx[edge_index]
    return edge_index, edge_attr


def get_global_proxy(local_proxy, classes, client_ids, train_label_count):
    global_proxy = []
    for c in classes:
        label_counts = [train_label_count[client_id][c] for client_id in client_ids]
        proxy_sum = sum([(local_proxy[client_id][c] * label_counts[client_id]) for client_id in client_ids])
        proxy_mean = proxy_sum / sum(label_counts)
        global_proxy.append(proxy_mean)
    return global_proxy