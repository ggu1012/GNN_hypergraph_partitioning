#!/usr/bin/env python

import torch

# x_coo = torch.tensor([[1,0,0], [1,1,0], [1,0,1], [0,1,1]]).to_sparse_coo()
x_coo = torch.load('/home/shkim/gnn_partitioning/dataset/ISPD98_benchmark_suite/ibm18/ibm18_coo.pt')
x_coo = x_coo.coalesce()


ind = x_coo.indices()
srt = x_coo.indices()[:,x_coo.indices().sort(1)[1][1]]
row_change_loc = torch.argwhere(torch.diff(ind[0]))[:,0] + 1
col_change_loc = torch.argwhere(torch.diff(srt[1]))[:,0] + 1
_len = srt.shape[1] * torch.ones(1, dtype=int)
_zero = torch.zeros(1, dtype=int)
_diff_r = torch.cat([_zero, row_change_loc, _len])
_diff_c = torch.cat([_zero, col_change_loc, _len])
_split_r = torch.diff(_diff_r).to(torch.int).tolist()
_split_c = torch.diff(_diff_c).to(torch.int).tolist()

drd = torch.split(ind[1] ,_split_r)
dcd = torch.split(srt[0], _split_c)

# print(drd)
print(ind[:,0:30])
print(srt[:,0:30])
print(dcd[4])
print(ind[:,torch.argwhere(ind[1] == 4).flatten()])
