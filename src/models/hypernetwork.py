import torch
import torch.nn as nn

from typing import List
from torch.nn import functional as F

def init_weights_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.01)

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hid_dim: int, 
                 out_dim:int, 
                 device: torch.device,
                 init_way: str = 'xavier') -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.net.to(device)

        if init_way == 'xavier':
            self.net.apply(init_weights_xavier)
        elif init_way == 'normal':
            self.net.apply(init_weights_normal)

    def forward(self, feature):
        return self.net(feature)


class MLP3(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hid_dim: int, 
                 out_dim:int, 
                 device: torch.device,
                 init_way: str = 'xavier') -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.net.to(device)

        if init_way == 'xavier':
            self.net.apply(init_weights_xavier)
        elif init_way == 'normal':
            self.net.apply(init_weights_normal)

    def forward(self, feature):
        return self.net(feature)

class HyperNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            clients: List[int],
            hidden_dim: int,
            device: torch.device,
            init_way: str = 'xavier'
        ) -> None:

        super(HyperNetwork, self).__init__()

        self.clients = clients
        self.client_num = len(clients)
        self.mp = {clients[i]: i for i in range(self.client_num)}
        self.device = device
        self.embedding = nn.Embedding(self.client_num, embedding_dim, device=self.device)

        self.mlps = [MLP(embedding_dim, hidden_dim, self.client_num - 1, device, init_way)  for _ in  range(self.client_num)]

    def mlp_parameters(self, client_id: int) -> List[nn.Parameter]:
        current_client_id = self.mp[client_id]
        return list(filter(lambda p: p.requires_grad, self.mlps[current_client_id].parameters()))

    def emd_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def forward(self, client_id: int, activation: str = 'sigmoid'):
        current_client_id = self.mp[client_id]
        emd = self.embedding(
            torch.tensor(current_client_id, dtype=torch.long, device=self.device)
        )

        if activation == 'relu':
            alpha = F.relu(self.mlps[current_client_id](emd))
        elif activation == 'sigmoid':
            alpha = F.sigmoid(self.mlps[current_client_id](emd))
        
        return alpha
