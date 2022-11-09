#!/usr/bin/env python3

import os
import time
from utils import *
import torch
from torch import nn
from torch import optim
from torch_geometric.nn import HypergraphConv

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class HyperGAP(nn.Module):
    def __init__(
        self,
        p_num, # the number of partitions        
        conv_emb_size,
        part_emb_size
    ):
        '''
        args:
            inc_indice (tensor.shape == (2,-1)): the location(index) of 1 in incidence matrix
            p_num (int): the number of partitions
            layer_num (int): the number of GNNConv layers
            conv_emb_size (list): embedding sizes for gnn layers            
        '''
        super(HyperGAP, self).__init__()

        assert p_num == part_emb_size[-1], "the number of partition = MLP layer size [-1]"
        self.p_num = p_num
        self.gnn_layers = nn.ModuleList([HypergraphConv(conv_emb_size[n], conv_emb_size[n+1]) for n in range(len(conv_emb_size) - 1)])
        self.mlp_layers = nn.ModuleList([nn.Linear(part_emb_size[n], part_emb_size[n+1]) for n in range(len(part_emb_size) - 1)])
        self.hidden_act = nn.ReLU()
        self.final_act = nn.Softmax(dim=1)

    def forward(self, x, inc_idx):
        for hconv in self.gnn_layers:
            x = hconv(x, inc_idx)

        # now we have node embeddings
        # translate embs to probability with MLP
        for mlp in self.mlp_layers[:-1]:
            x = mlp(x)
            x = self.hidden_act(x)
        
        return self.final_act(self.mlp_layers[-1](x))     

    def loss(self, prob, inc_idx_nested):
        '''
        args:
            prob (torch.Tensor (v_num, p_num)): probability matrix, output from HyperGAP layers
            inc_idx_nested (torch.sparse_coo (v_num, e_num)): incidence matrix of hypergraph
        '''

        xfx = inc_idx_nested
        extended = torch.cat((prob, torch.zeros((1,prob.shape[1]), device='cuda')), 0)
        xdx = extended[xfx]
        print(xdx.shape)
        part_prob = 1 - (1 - xdx).prod(1)
        connectivity = part_prob.sum()

        part_expected_v_nums = prob.sum(dim=0)
        balancedness = torch.pow(part_expected_v_nums - prob.shape[0] // prob.shape[1], 2).sum()

        return connectivity, balancedness
    

if __name__ == '__main__':
    device = 'cuda'
    
    inc_mat = load_ISPD_benchmark('/home/shkim/gnn_partitioning/dataset', 1)


    x = torch.randn(inc_mat.indices()[0][-1] + 1, 256).to(device)
    dcd = split_col_inc_idx(inc_mat).to(device)
    # dcd = torch.load('/home/shkim/gnn_partitioning/dcd').to(device)
    partition_num = 2

    conv_emb_sz = [256, 128, 64]
    lin_emb_sz = [64, 16, 8, partition_num]
    model = HyperGAP(partition_num, conv_emb_sz, lin_emb_sz).to(device)

    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

    # optimizer = optim.SGD(params, lr=0.03, momentum=0.9)
    optimizer = optim.Adam(params, lr=0.03)
    _ind = inc_mat.indices().long().to(device)

    start = time.time()

    # init embs generation
    for ep in range(10):
        optimizer.zero_grad()
        prob = model.forward(x, _ind)
        conn, bal = model.loss(prob, dcd)
        loss = (conn - inc_mat.shape[1]) + 0.01 * bal

        loss.backward()
        optimizer.step()
        # print(f'({ep:>5}) loss: {loss.item():.3f}, connectivity: {conn.item() - inc_mat.shape[1]:.3f}, balance: {bal.item():.3f}')
        
        if ep % 200 == 0 :
            torch.save(model.state_dict(), "./model_dict")
