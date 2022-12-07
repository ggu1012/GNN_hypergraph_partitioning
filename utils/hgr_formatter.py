#!/usr/bin/env python3

import torch
from my_utils import *

def hgr_conversion(inc_mat: torch.sparse_coo):
    split_col, _ = split_col_inc_idx(inc_mat, padded=0)

    with open('./compressed.hgr', 'w') as hgr:
        hgr.write(f'{inc_mat.shape[1]} {inc_mat.shape[0]}\n')
        for edge in split_col:
            for v in edge[:-1]:
                hgr.write(f"{v+1} ")
            hgr.write(f"{edge[-1]+1}\n")
        
if __name__ == '__main__':
    hgr_conversion(torch.load('../ex.pt').coalesce())
