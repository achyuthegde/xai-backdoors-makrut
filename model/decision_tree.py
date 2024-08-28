from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from base import base_model

class Tree():
    def __init__(self):
        self.num_cut = [1, 1]
        self.num_leaf = np.prod(np.array(self.num_cut) + 1)
        self.num_class = 2
        self.model = nn.Sequential(
            nn.Linear(31,16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.Sigmoid()
            )
        """
        self.model = nn.Sequential(
            nn.Linear(31,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2),
            nn.Sigmoid()
            )
        """ 
    def torch_kron_prod(self, a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res
    
    def torch_bin(self, x, cut_points, temperature=0.1):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0),0)
        h = torch.matmul(x, W) + b
        res = torch.exp(h-torch.max(h))
        res = res/torch.sum(res, dim=-1, keepdim=True)
        return h
    
    def nn_decision_tree(self, x, cut_points_list, leaf_score, temperature=0.1):
        leaf = reduce(self.torch_kron_prod,
                      map(lambda z: self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
        return torch.matmul(leaf, leaf_score)

    