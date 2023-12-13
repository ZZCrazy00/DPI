import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


def full_attention_conv(qs, ks, vs):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    all_ones = torch.ones([vs.shape[0]]).to(vs.device)
    vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
    attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]
    return attn_output


def gcn_conv(x, edge_index):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append(matmul(adj, x[:, i]))  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1)  # [N, H, D]
    return gcn_conv_output


class TransformerConv(nn.Module):
    def __init__(self, channels, num_heads, use_graph=True):
        super(TransformerConv, self).__init__()
        self.Wk = nn.Linear(channels, channels * num_heads)
        self.Wq = nn.Linear(channels, channels * num_heads)
        self.Wv = nn.Linear(channels, channels * num_heads)

        self.out_channels = channels
        self.num_heads = num_heads
        self.use_graph = use_graph

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        attention_output = full_attention_conv(query, key, value)  # [N, H, D]
        if self.use_graph:
            final_output = attention_output + gcn_conv(value, edge_index)
        else:
            final_output = attention_output
        final_output = final_output.mean(dim=1)

        return final_output


class my_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(my_Transformer, self).__init__()
        num_layers = args.dm_layers
        num_heads = args.dm_heads
        self.residual = args.dm_residua
        self.use_graph = args.dm_graph

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(TransformerConv(hidden_channels, num_heads, self.use_graph))
            self.bns.append(nn.LayerNorm(hidden_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.bns[0](self.fcs[0](x))), p=0.2)
        layer_ = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index)
            if self.residual:
                x = 0.5 * x + 0.5 * layer_[i]
                x = self.bns[i + 1](x)
            x = F.dropout(x, p=0.2, training=self.training)
            layer_.append(x)
        x_out = self.fcs[-1](x)
        return x_out
