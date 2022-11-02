# hypergraph generation
# from netlist statistics

import torch
import random


def analysis(netlist_coo_pt):
    hgraph_coo = torch.load(netlist_coo_pt)
    # coo tensor indice format: [node, hedge]

    pin_indices = hgraph_coo._indices()[0, :]
    hedge_indices = hgraph_coo._indices()[1, :]

    pin_list = pin_indices.unique(return_counts=True)[1]
    fanout_list = hedge_indices.unique(return_counts=True)[1]

    # hyperedge net weight
    x = fanout_list.unique(return_counts=True)
    y = pin_list.unique(return_counts=True)
    edge_sizes, edge_prob = x[0], x[1] / x[1].sum()
    pin_sizes, pin_prob = y[0], y[1] / y[1].sum()

    return (edge_sizes, edge_prob), (pin_sizes, pin_prob)


def generic_edge_pin_sizes(E_num, V_num, edge_stat, pin_stat):

    edge_sizes = edge_stat[0]
    e_prob = edge_stat[1]
    pin_sizes = pin_stat[0]
    p_prob = pin_stat[1]

    # hedge sizes, pin sizes
    hedge_sz_idx = torch.multinomial(e_prob, E_num, replacement=True)
    sampled_hedge_sz = edge_sizes[hedge_sz_idx]
    pin_sz_idx = torch.multinomial(p_prob, V_num, replacement=True)
    sampled_pin_sz = pin_sizes[pin_sz_idx]

    hedge_total_sz = sampled_hedge_sz.sum()
    pin_total_sz = sampled_pin_sz.sum()

    diff = hedge_total_sz - pin_total_sz
    if diff > 0:
        # pick indices to be added by 1
        picked_idx = torch.randperm(V_num)[:diff]
        sampled_pin_sz[picked_idx] += 1

    elif diff < 0:
        # pick indices to be added by 1
        picked_idx = torch.randperm(E_num)[: abs(diff)]
        sampled_hedge_sz[picked_idx] += 1

    sorted_pin_sz = sampled_pin_sz.sort()[0]
    sorted_hedge_sz = sampled_hedge_sz.sort()[0]

    return sorted_pin_sz, sorted_hedge_sz


class sp_matrix:
    def __init__(self, pin_sizes, hedge_sizes):

        self.pin_sizes = pin_sizes
        self.hedge_sizes = hedge_sizes

    def __init_mat(self):
        E_filled = torch.zeros(self.hedge_sizes.shape)
        V_available_tail_len = (self.hedge_sizes.shape[0] - 1) * torch.zeros(
            self.pin_sizes.shape
        )

        hedge_num = self.hedge_sizes.shape[0]

        self.col = torch.arange(self.pin_sizes[0])
        E_filled[0 : self.pin_sizes[0]] += 1
        V_available_tail_len[0] = hedge_num - self.pin_sizes[0]
        for pin_idx, num_of_pins in enumerate(self.pin_sizes[1:]):
            start_idx = torch.min(torch.argwhere(E_filled < self.hedge_sizes).flatten())
            end_idx = start_idx + num_of_pins
            if end_idx >= hedge_num:
                self.col = torch.cat(
                    (self.col, torch.arange(hedge_num - num_of_pins, hedge_num))
                )
                E_filled[hedge_num - num_of_pins : hedge_num] += 1
                V_available_tail_len[pin_idx + 1] = 0
            else:
                self.col = torch.cat((self.col, torch.arange(start_idx, end_idx)))
                E_filled[start_idx:end_idx] += 1
                V_available_tail_len[pin_idx + 1] = hedge_num - end_idx

        print(V_available_tail_len)
        self.__gen_infos = (E_filled, V_available_tail_len)

    def __refine_mat(self):
        start_idx = 0
        for i in self.pin_sizes:
            one_row_last = start_idx + i - 1
            if (one_row_last > 3):
                x = 1

    def generate_mat(self):

        self.__init_mat()
        self.__refine_mat()


if __name__ == "__main__":

    base_path = "/home/shkim/gnn_partitioning/dataset/ISPD98_benchmark_suite/"
    bench = 16
    test = 1

    if not test:
        e_stat, p_stat = analysis(base_path + f"ibm{bench:02d}/ibm{bench:02d}_coo.pt")
        xx, yy = generic_edge_pin_sizes(200, 200, edge_stat=e_stat, pin_stat=p_stat)
    else:
        xx = torch.Tensor([1, 6, 9, 4, 3, 3, 5, 2, 2, 6]).to(torch.int)
        yy = torch.Tensor([3, 3, 3, 3, 4, 4, 4, 4, 6, 7]).to(torch.int)

    a = sp_matrix(xx, yy)
    a.generate_mat()

