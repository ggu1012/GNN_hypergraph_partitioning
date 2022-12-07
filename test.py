#!/usr/bin/env python3

import os
import time
from model.model import *
from utils.my_utils import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda'

model = torch.load('./model_dict', map_location=device)

# init_feat = torch.load('./init_embs_ibm01_256')
init_feat = torch.load('./xxxxx')
inc_mat = torch.load('./dataset/tensored/ISPD_ibm01_coo.pt').coalesce()

with torch.no_grad():
    model.eval()
    prob = model(init_feat.to(device), inc_mat.to(device))

prob = prob.detach().cpu()
_, label = torch.max(prob, dim=1)

part_nums = label.unique(return_counts = True)[1]
expect = prob.sum(dim=0)
print(part_nums)
print(expect)
print(f'bal: {torch.min(part_nums)/torch.max(part_nums)}')
print(f'cont_bal: {torch.min(expect)/torch.max(expect)}')

xfx, _ = split_col_inc_idx(inc_mat, padded=0)
tft = xfx.to_padded_tensor(label.shape[0]).to(device)

divided_edges = 0
conn = 0
for hedge in xfx:
    one_edge_part_num = label[hedge].unique()
    conn += (one_edge_part_num.shape[0] - 1)
    if (one_edge_part_num.shape[0] - 1) > 0:
        divided_edges += 1
print(f'conn: {conn}, div_edges: {divided_edges}')

with open('./dataset/hgr_benchmark_set/ISPD98_ibm01.hgr.part.4') as f:
    x = f.readlines()
    hmetis_result = torch.tensor([int(y.strip()) for y in x])

divided_edges = 0
hmetis_conn = 0
for hedge in xfx:
    one_edge_part_num = hmetis_result[hedge].unique()
    # print(one_edge_part_num)
    hmetis_conn += (one_edge_part_num.shape[0] - 1)
    if (one_edge_part_num.shape[0] - 1) > 0:
        divided_edges += 1
print(f'hmetis_conn: {hmetis_conn}, div_edges: {divided_edges}')


p_num = prob.shape[1]
v_num = init_feat.shape[0]
shuffled = torch.randperm(v_num)
random_labels = torch.zeros(v_num)
one_size = v_num // p_num
i=0
for x in range(p_num):
    j = i+one_size if i+one_size < v_num else -1
    random_labels[shuffled[i:j]] = x
    i += one_size

random_conn = 0
divided_edges = 0
for hedge in xfx:
    one_edge_part_num = random_labels[hedge].unique()
    parts = one_edge_part_num.shape[0] - 1
    random_conn += parts
    if parts > 0:
        divided_edges += 1
print(f'random_conn: {random_conn}, div_edges: {divided_edges}')


prob = prob.cuda()
extended = F.pad(prob, (0, 0, 0, 1), "constant", 0)
xdx = extended.index_select(dim=0, index=tft.view(-1)).view((tft.shape[0], -1, prob.shape[1]))
part_prob = 1 - (1 - xdx).prod(1)
print(part_prob.sum() - tft.shape[0])