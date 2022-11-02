#!/usr/bin/env python3

from re import I
import torch
from torch import nn

class GNNConv(nn.Module):
    def __init__(self, in_out):
        '''
        args:
            in_out (tuple): (input_size, output_size)
        '''
        super(GNNConv, self).__init__()
        self.layer = nn.Linear(2 * in_out[0], in_out[1], bias=False)
        self.in_out = in_out
        self.act = nn.ReLU()

        for param in self.parameters():
            nn.init.kaiming_normal_(param)

        def forward(self, in_feat, agg_feat):
            '''
            args:
                in_feat (torch.Tensor): feature of current node
                agg_feat (torch.Tensor): aggregated feature from neigh's nodes
            '''
            concat = torch.cat((in_feat, agg_feat), dim=0).float().view(-1,1).t()
            output = self.act(self.layer(concat).t()).flatten()
            return output 

class HyperGAP(nn.Module):
    def __init__(
        self,
        p_num, # the number of partitions    
        adj_map
    ):

        self.v_num = adj_map.shape[0]
        self.e_num = adj_map.shape[1]
        self.p_num = p_num

    def cut_loss(self, prob, adj_mat):
        '''
        args:
            prob (torch.Tensor (v_num, p_num)): probability matrix, output from HyperGAP layers
            adj_mat (torch.sparse_csr (v_num, e_num)): adjacent matrix of hypergraph
        '''
        p_num = self.p_num
        v_num = self.v_num
        e_num = self.e_num
        assert prob.shape == (v_num, p_num) # v x p
        
        prob_3dim = prob.T.view((p_num, v_num, -1)) # p x v x 1
        
        prob_p_e = torch.empty((p_num, e_num)) # p x e
        for part in range(p_num):
            # TODO: Sparse-Dense Element-wise multiplication

        return E_cut


def load_ISPD_benchmark(dataset_path, bench_num):
    '''
    args:
        dataset_path (string): dataset pwd
        bench_num (int): ISPD98 benchmark circuit number, 1~18
    return:
        adj (torch.sparse_csr): Hypergraph adjacency matrix with torch sparse csr format 
    '''
    adj = torch.load(dataset_path + f'/ISPD98_benchmark_suite/ibm{bench_num:02d}/ibm{bench_num:02d}_adj.pt')
    return adj 

if __name__ == '__main__':
    xj = load_ISPD_benchmark('/home/shkim/gnn_partitioning/dataset', 1)
    print(xj) 