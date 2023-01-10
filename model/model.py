#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import torch_geometric.nn as gnn

from utils.my_utils import *


class HyperGAP(nn.Module):
    def __init__(
        self,
        p_num,  # the number of partitions
        conv_emb_size,
        part_emb_size,
        gnn_dropout=0.5,
        mlp_dropout=0.5,
    ):
        """
        args:
            inc_indice (tensor.shape == (2,-1)): the location(index) of 1 in incidence matrix
            p_num (int): the number of partitions
            layer_num (int): the number of GNNConv layers
            conv_emb_size (list): embedding sizes for gnn layers
        """
        super(HyperGAP, self).__init__()

        assert (
            p_num == part_emb_size[-1]
        ), "the number of partition = MLP layer size [-1]"
        self.p_num = p_num

        self.gnn_layers = nn.ModuleList(
            [
                HypergraphConv(
                    in_channels=conv_emb_size[n],
                    out_channels=conv_emb_size[n + 1],
                    dropout=gnn_dropout,
                )
                for n in range(len(conv_emb_size) - 1)
            ]
        )
        self.gnn_norm = nn.ModuleList(
            [gnn.GraphNorm(conv_emb_size[n + 1]) for n in range(len(conv_emb_size) - 1)]
        )
        self.gnn_act = nn.LeakyReLU()
        self.gnn_drop = nn.Dropout(gnn_dropout)
        self.mlp = gnn.models.MLP(
            part_emb_size, dropout=mlp_dropout, act="LeakyReLU", norm="InstanceNorm"
        )
        self.final_act = nn.Softmax(dim=1)

    def forward(self, x, inc_mat):
        inc_idx = inc_mat.indices().long()
        for i, hconv in enumerate(self.gnn_layers):
            x = hconv(x=x, hyperedge_index=inc_idx)
            x = self.gnn_norm[i](x)
            x = self.gnn_act(x)
            x = self.gnn_drop(x)
        x = self.mlp(x)
        # x = self.final_act(x)
        x = F.gumbel_softmax(x, tau=0.1, hard=False)
        return x

    def loss(self, prob, inc_idx_nested, hedge_sz, vol_V, device):
        """
        args:
            prob (torch.Tensor (v_num, p_num)): probability matrix, output from HyperGAP layers
            inc_idx_nested (torch.sparse_coo (v_num, e_num)): incidence matrix of hypergraph
        """
        v_num, p_num = prob.shape
        e_num = inc_idx_nested.shape[0]

        xfx = inc_idx_nested
        extended = F.pad(prob, (0, 0, 0, 1), "constant", 0)

        # _, label = torch.max(prob, dim=1)
        # label = torch.cat([label, torch.tensor([-1]).cuda()])
        # discrete_label = label.index_select(dim=0, index=xfx.view(-1)).view(xfx.shape)
        # discrete_conn = 0
        # for x in discrete_label:
        #     discrete_conn += (x.unique().shape[0]-2)

        #### indexing tweak.
        # use index_select for fast indexing
        # view(-1) for mat-to-vec
        # view(.,-1,.) to reshape
        # works as 'extended[xfx]'
        xdx = extended.index_select(dim=0, index=xfx.view(-1)).view((e_num, -1, p_num))
        # xdx.shape = (e, -1, p)
        # -1 means the maximum number of non-zero values in one hyperedge
        ###

        part_prob = 1 - (1 - xdx).prod(1)

        # Connectivity; (lambda-1) metrics
        connectivity = part_prob.sum()

        _connectivity = part_prob.sum(1)
        weighted_cut = _connectivity / hedge_sz
        w_connectivity = weighted_cut.sum()

        # normalized cut
        vol_S = xdx.sum(dim=1).sum(dim=0)
        vol_Sinv = vol_V - vol_S
        ww = (vol_V / (vol_S * vol_Sinv)) * (
            1 - torch.eye(prob.shape[1], device=device)
        )
        zz = ww @ part_prob.T
        cut_S = part_prob.T.mul(zz).sum()

        part_expected_v_nums = prob.sum(dim=0)
        # balance
        balancedness = torch.pow(part_expected_v_nums - v_num // p_num, 2).sum() / p_num
        # negative entropy for balance
        # p = (part_expected_v_nums) / v_num
        # print(p)
        # entrpy =  (p * torch.log2(p)).sum()

        return (
            connectivity,
            cut_S,
            balancedness,
            w_connectivity,
            _connectivity,
        )
    
    def gumbel_loss(self, prob, inc_idx_nested, device):
        v_num, p_num = prob.shape
        e_num = inc_idx_nested.shape[0]

        xfx = inc_idx_nested
        extended = F.pad(prob, (0, 0, 0, 1), "constant", 0)

        #### indexing tweak.
        # use index_select for fast indexing
        # view(-1) for mat-to-vec
        # view(.,-1,.) to reshape
        # works as 'extended[xfx]'
        xdx = extended.index_select(dim=0, index=xfx.view(-1)).view((e_num, -1, p_num))
        # xdx.shape = (e, -1, p)
        # -1 means the maximum number of non-zero values in one hyperedge
        ###

        connectivity = xdx.max(dim=1).values.sum()
        balancedness = torch.pow(prob.sum(dim=0) - v_num // p_num, 2).sum() / p_num

        return connectivity, balancedness