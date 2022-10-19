#!/usr/bin/env python3

import torch
from torch import nn

# class GNNConv(nn.Module):



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