#!/usr/bin/env python3

from utils import *
import torch
import numpy as np
from torch import nn

class SAGEConv(nn.Module):
    def __init__(self, in_size, out_size):
        '''
        args:
            in_out (tuple): (input_size, output_size)
        '''
        super(SAGEConv, self).__init__()
        self._layer = nn.Linear(2 * in_size, out_size, bias=False)
        self.in_size = in_size
        self.act = nn.LeakyReLU()

        for param in self.parameters():
            nn.init.kaiming_normal_(param)

        def forward(self, in_feat, agg_feat):
            '''
            args:
                in_feat (torch.Tensor): feature of current node
                agg_feat (torch.Tensor): aggregated feature from neigh's nodes
            '''

            assert in_size == in_feat.shape[0] , "input feature size is incompatible"
            concat = torch.cat((in_feat, agg_feat), dim=0).float().view(-1,1).t()
            output = self.act(self._layer(concat).t()).flatten()
            return output 

class HyperGAP(nn.Module):
    def __init__(
        self,
        p_num, # the number of partitions        
        GNNConv,
        layer_num,
        emb_sizes,
        init_embs
    ):
        '''
        args:
            inc_indice (tensor.shape == (2,-1)): the location(index) of 1 in incidence matrix
            p_num (int): the number of partitions
            GNNConv (class): torch GNN model
            layer_num (int): the number of GNNConv layers
            emb_sizes (list): embedding sizes for gnn layers            
        '''
        super(HyperGAP, self).__init__()
        
        assert layer_num + 1 != len(emb_sizes), "check layer_num and emb_sizes"

        self.p_num = p_num
        self.init_embs = init_embs

        self.gnn_layers = [GNNConv(emb_sizes[n], emb_sizes[n+1]) for n in range(layer_num)]
        self.mlp_layer = nn.Linear(emb_sizes[-1], 2)
  
    def _aggregate(self, batch_idx, adj_mat):
        # sample neighbor nodes
        
        


    def _loss_fnc(self, prob, inc_mat):
        '''
        args:
            prob (torch.Tensor (v_num, p_num)): probability matrix, output from HyperGAP layers
            inc_mat (torch.sparse_coo (v_num, e_num)): incidence matrix of hypergraph
        '''
        p_num = self.p_num
        v_num = self.v_num
        e_num = self.e_num
        assert prob.shape == (v_num, p_num) # v x p
        

        return E_cut
    

        





# if __name__ == '__main__':
#     inc_mat = load_ISPD_benchmark('/home/shkim/gnn_partitioning/dataset', 1)
    

