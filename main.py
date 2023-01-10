#!/usr/bin/env python3

import os
import time
from model.model import *
from utils.my_utils import *
import torch
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda'

bench = '01'
partition_num = 2


inc_mat = load_tensored_hgr(f'/home/shkim/gnn_partitioning/dataset/tensored/ISPD_ibm{bench}_coo.pt')

init_emb_size = 256
x = torch.randn(inc_mat.indices()[0][-1] + 1, init_emb_size).to(device)
torch.save(x, './xxxxx')
# x = torch.load(f'./init_embs_ibm{bench}_256').to(device)
init_emb_size = x.shape[1]

dcd, hsz = split_col_inc_idx(inc_mat, padded=1)
inc_mat = inc_mat.to(device)
dcd, hsz = dcd.to(device), hsz.to(device)

conv_emb_sz = [init_emb_size, 64, 16]
lin_emb_sz = [16, partition_num]
# lin_emb_sz = [init_emb_size, 64, partition_num]

model = HyperGAP(partition_num, conv_emb_sz, lin_emb_sz, gnn_dropout=0.5, mlp_dropout=0.6).to(device)

params = []
for param in model.parameters():
    if param.requires_grad:
        params.append(param)

# optimizer = optim.SGD(params, lr=0.001, momentum=0.1)
optimizer = optim.Adam(params, lr=5e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
vol_V = inc_mat.indices().shape[1]

# init embs generation
start_time = time.time()
loss_ = []
cut_ = []
conn_ = []
bal_ = []
sxd_ = []

plt.figure(figsize=(10,10))
diff_ = []


for ep in range(10000):
    optimizer.zero_grad()
    prob = model.forward(x, inc_mat)
    conn, cut, bal, w_conn, _conn = model.loss(prob, dcd, hsz, vol_V, device)

    conn, bal = model.gumbel_loss(prob, dcd, device)

    # loss = cut

    # loss = (10 * cut + 1e-3 * bal)
    loss = (conn + 1e-3 * bal)
    # loss = conn + cut

    loss.backward()
    optimizer.step()
    print(f'({ep:>5}) loss: {loss.item():.3f}, cut: {cut.item():.3f}, conn: {conn.item() - inc_mat.shape[1]:.3f}, balance: {bal.item():.3f}')
    
    loss_.append(loss.item())
    cut_.append(cut.item())
    conn_.append(conn.item() - inc_mat.shape[1])
    bal_.append(bal.item())
    # plt.figure()
    # plt.hist(d1.detach().cpu())
    # plt.hist(d2)
    # print(d2)
    # plt.savefig('./yyy.png')   

    tmp = prob.detach().cpu().numpy()    
    if ep == 0:
        old = tmp.argmax(axis=1)    
    new = tmp.argmax(axis=1)
    diff_.append(np.argwhere((new - old) != 0).flatten().shape[0])
    if (ep) % 100 == 0:
        pd.DataFrame(tmp).to_csv('tmp.csv')
        pd.DataFrame(np.abs((new - old))).to_csv('tmp1.csv')   
    old = new

    if (ep) % 200 == 0:
        torch.save(model, "./model_dict")        
        plt.clf()
        plt.subplot(3,2,1).set_title('loss')
        plt.plot(loss_)
        plt.yscale('log')
        plt.subplot(3,2,2).set_title('normalized cut')
        plt.plot(cut_)
        plt.subplot(3,2,3).set_title('E(connectivity)')
        plt.plot(conn_)
        plt.subplot(3,2,4).set_title('balance')
        plt.plot(bal_)
        plt.yscale('log')
        plt.subplot(3,2,5).set_title('swapped vertices')
        plt.plot(diff_)
        plt.savefig('cut.png')
        print(prob.sum(0))
        print(f'unbal={torch.min(prob.sum(0))/torch.max(prob.sum(0))}')

        
with torch.no_grad():
    # model.eval()
    prob = model.forward(x, inc_mat)
    print(prob.sum(0))


print(time.time() - start_time)
