import torch

def load_tensored_hgr(dataset_path):
    '''
    args:
        dataset_path (string): tensored pwd
    return:
        inc.indice() (torch.tensor (2, *))): Location(index) of 1 in hypergraph incidence matrix
    '''
    inc = torch.load(dataset_path)
    return inc.coalesce()


def split_col_inc_idx(inc_mat, padded):

    ## split by col
    inc_idx = inc_mat.indices()
    srt = inc_idx[:, inc_idx.sort(1)[1][1]]
    col_change_loc = torch.argwhere(torch.diff(srt[1]))[:,0] + 1
    _len = srt.shape[1] * torch.ones(1) # for diff
    _zero = torch.zeros(1) # for diff
    _diff_ = torch.cat([_zero, col_change_loc, _len])
    hedge_sz = torch.diff(_diff_).to(torch.int) # split_sz = hedge_sz of each hedge

    _split_sz = torch.split(srt[0], hedge_sz.tolist()) 
    _nested_inc = torch.nested.nested_tensor(list(_split_sz)) # Pytorch 1.13 torch.nested.nested_tensor(...)
    xfx = _nested_inc.to_padded_tensor(inc_mat.shape[0]) if padded else _nested_inc
    
    return xfx, hedge_sz


# def evaluate(prob, inc_mat):



# def inc_to_degadj(inc_mat):
#     '''
#     return: adjacency matrix with degree info.
#             used for aggregation step
#     '''
    

#     return adj_mat_coo



# def batch_sampling(inc_mat):
#     '''
#     sampling standard: pick 'edge'
#     '''

    
