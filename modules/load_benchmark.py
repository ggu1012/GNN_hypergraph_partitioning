#!/usr/bin/env python

import numpy as np
import torch

def load_benchmark(path):
    '''
    args:
        path (string): absolute path to ISPD98 benchmark net file
    
    output:
        adj_mat (np.array): hypergraph in format of adjacency matrix
    '''

    with open(path, 'r') as ckt:
        # header 5 lines
        ckt.readline()
        num_pins = int(ckt.readline())
        num_nets = int(ckt.readline())
        num_modules = int(ckt.readline())
        num_cells = int(ckt.readline())
        
        cell_idx = dict()
        adj_mat = np.zeros((num_nets, num_modules), dtype=bool)
        # initialized size = e x v
        # should be transposed later

        net_num = -1
        cell_num = 0 

        while True:
            line = ckt.readline()
            if not line:
                break
            cmps = line.strip().split(' ')
            if cmps[1] == 's':
                net_num += 1
            if cmps[0] not in cell_idx.keys():
                cell_idx[cmps[0]] = cell_num
                cell_num += 1 
            adj_mat[net_num][cell_idx[cmps[0]]] = 1

    assert net_num == num_nets - 1
    assert cell_num == num_modules

    return adj_mat.T

def main():
    base_path = '/home/shkim/gnn_partitioning/dataset/ISPD98_benchmark_suite/'
    for bench in range(1, 19):
        print(f'Converting ibm{bench:02d}.net...')
        adj = load_benchmark(base_path + f'ibm{bench:02d}/ibm{bench:02d}.net')
        # np.save(base_path + f'ibm{bench:02d}/ibm{bench:02d}_adj.npy', adj) 
        adj = torch.from_numpy(adj).to_sparse_csr()
        torch.save(adj, base_path+f'ibm{bench:02d}/ibm{bench:02d}_adj.pt')
if __name__=='__main__':
    main()