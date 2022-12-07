#!/usr/bin/env python

import torch
import argparse

def load_hgr(path):
    '''
    args:
        path (string): absolute path to ISPD98 benchmark net file
    
    output:
        inc_mat (torch.sparse.coo_matrix): hypergraph in format of incidence matrix, sparse coo format
    '''

    with open(path, 'r') as ckt:
        # header line
        line = ckt.readline().strip()
        e_num = int(line.split(' ')[0])
        v_num = int(line.split(' ')[1])
        
        line_num = 0
        row, col = [], []
        while True:
            line = ckt.readline()
            if not line:
                break
            line = [int(x) for x in line.strip().split(' ')]
            for idx in line:
                row.append(line_num)
                col.append(idx - 1)
            line_num += 1
        
        assert len(row) == len(col)
        val = [1 for _ in range(len(row))]
        ind = [row, col]
        inc_mat = torch.sparse_coo_tensor(ind, val, size=(e_num, v_num)).to(torch.int)


    return inc_mat.T

def main():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--gen', '-g', dest="gen", action="store_true")
    parser.add_argument('--num', '-n', dest="num", action="store")
    args = parser.parse_args()

    base_path = '/home/shkim/gnn_partitioning/dataset/hgr_benchmark_set/'


    if not args.gen:
        for bench in range(1, 19):
            print(f'Converting ibm{bench:02d}.net...')
            inc = load_hgr(base_path + f'ISPD98_ibm{bench:02d}.hgr')
            print(inc)
            torch.save(inc, base_path+f'../tensored/ISPD_ibm{bench:02d}_coo.pt')
    else:
        print("Extracting data for generic hypergraph")
        bench = args.num
        adj = load_hgr(base_path + f'ibm{bench:02d}/ibm{bench:02d}.net')


if __name__=='__main__':
    main()