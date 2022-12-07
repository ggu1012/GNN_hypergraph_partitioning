#!/usr/bin/env python3

from utils.my_utils import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigsh
import numpy as np
import torch

inc_mat = load_tensored_hgr('/home/shkim/gnn_partitioning/dataset/tensored/ISPD_ibm18_coo.pt')
inc_mat_ = coo_matrix((inc_mat.values(), inc_mat.indices()), shape=inc_mat.shape, dtype=np.float32)

# _idx = ([0,1,1,2,2,3,3], [0,0,1,0,2,1,2])
# _val = [1 for _ in range(len(_idx[0]))]
# inc_mat_ = coo_matrix((_val, _idx), shape=(_idx[0][-1]+1, _idx[1][-1]+1), dtype=np.float32)


info_mat = np.dot(inc_mat_, inc_mat_.T)


# u, s, vh = svds(coo_matrix(info_mat), k=2)
# print(u)
# t_u = torch.from_numpy(u)
# torch.save(t_u, './ex_embs')

w, v = eigsh(info_mat, 256)


print(w, v)

