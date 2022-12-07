#! /usr/bin/env python

# hypergraph generation
# from netlist statistics

import torch
import numpy as np


def analysis(netlist_coo_pt):
    hgraph_coo = torch.load(netlist_coo_pt)
    # coo tensor indice format: [node, hedge]

    pin_indices = hgraph_coo._indices()[0, :].numpy()
    hedge_indices = hgraph_coo._indices()[1, :].numpy()

    pin_list = np.unique(pin_indices, return_counts=True)[1]
    fanout_list = np.unique(hedge_indices,return_counts=True)[1]

    # hyperedge net weight
    x = np.unique(fanout_list, return_counts=True)
    y = np.unique(pin_list, return_counts=True)
    edge_sizes, edge_prob = x[0], (x[1] / x[1].sum())
    pin_sizes, pin_prob = y[0], (y[1] / y[1].sum())

    return (edge_sizes, edge_prob), (pin_sizes, pin_prob)


def generic_edge_pin_sizes(V_num, E_num, edge_stat, pin_stat):

    edge_sizes = edge_stat[0]
    e_prob = edge_stat[1]
    pin_sizes = pin_stat[0]
    p_prob = pin_stat[1]

    rng = np.random.default_rng()

    # hedge sizes, pin sizes
    sampled_hedge_sz = rng.choice(edge_sizes, E_num, p=e_prob)    
    sampled_pin_sz = rng.choice(pin_sizes, V_num, p=p_prob)
    hedge_total_sz = sampled_hedge_sz.sum()
    pin_total_sz = sampled_pin_sz.sum()

    diff = (hedge_total_sz - pin_total_sz).item()
    if diff > 0:
        # pick indices to be added by 1
        picked_idx = (rng.permutation(V_num * diff)[:diff]) % V_num
        x, y = np.unique(picked_idx, return_counts=True)
        sampled_pin_sz[x] += y

    elif diff < 0:
        # pick indices to be added by 1
        picked_idx = rng.permutation(E_num * abs(diff))[: abs(diff)] % E_num
        x, y = np.unique(picked_idx, return_counts=True)
        sampled_hedge_sz[x] += y

    sampled_pin_sz[::-1].sort()
    sampled_hedge_sz.sort()

    return sampled_pin_sz, sampled_hedge_sz


def generate_matrix(pin_sizes, hedge_sizes):
    E_filled = np.zeros(hedge_sizes.shape)
    hedge_num = hedge_sizes.shape[0]

    col = np.arange(pin_sizes[0])
    row = np.zeros(pin_sizes[0])
    E_filled[col] += 1

    # initial result
    for pin_idx, num_of_pins in enumerate(pin_sizes[1:]):
        start_idx = np.min(np.argwhere(E_filled < hedge_sizes).flatten())
        if start_idx + num_of_pins >= hedge_num:
            start_idx = hedge_num - num_of_pins
        end_idx = start_idx + num_of_pins
        _row = np.arange(start_idx, end_idx)

        # print(pin_idx)
        E_filled[_row] += 1
        col = np.concatenate((col, _row))
        row = np.concatenate((row, (pin_idx+1) * np.ones(num_of_pins)))

    #shuffle row, col
    rng = np.random.default_rng()
    ind = []
    for x in [row, col]:
        val, idx = np.unique(x, return_inverse=True)    
        shuffled_val = rng.permutation(val) 
        ind.append(shuffled_val[idx])

    return np.array(ind, dtype=int)

if __name__ == "__main__":

    base_path = "/home/shkim/gnn_partitioning/dataset/"
    bench = 1
    test = 0

    e_stat, p_stat = analysis(base_path + f"tensored/ISPD_ibm{bench:02d}_coo.pt")
    xx, yy = generic_edge_pin_sizes(1200, 1400, edge_stat=e_stat, pin_stat=p_stat)

    _idx = generate_matrix(xx, yy)
    _val = np.ones(_idx.shape[1], dtype=int)

    idx, val = torch.from_numpy(_idx), torch.from_numpy(_val)

    sp_mat = torch.sparse_coo_tensor(idx, val).coalesce()
    torch.save(sp_mat, '../ibm01_compressed_1200_1400.pt')

