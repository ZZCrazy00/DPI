import pandas as pd
import numpy as np
import torch
import torch_sparse
from pandas.util import hash_array
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from datasketch import HyperLogLogPlusPlus, hyperloglog_const


def get_src_dst_degree(src, dst, A, max_nodes):
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree


class MyDataset(Dataset):
    def __init__(self, args):
        super(MyDataset, self).__init__()
        self.data = self.process_data(args=args)

    def process_data(self, args):
        if args.dataset_name in ["BindingDB", "davis"]:
            adj = pd.read_excel("dataset/{}/dti_mat.xlsx".format(args.dataset_name), header=None)  # 9166 7320
            adj = adj.values.tolist()
            edge_index = []
            for i in range(len(adj)):
                for j in range(len(adj[0])):
                    if adj[i][j] == 1:
                        edge_index.append([i, j + len(adj)])
            edge_index = torch.LongTensor(edge_index).T

            drugs_feature = np.loadtxt("dataset/{}/drugs_feature.txt".format(args.dataset_name), delimiter=",").tolist()    # 3400 68
            targets_feature = np.loadtxt("dataset/{}/targets_feature.txt".format(args.dataset_name), delimiter=",").tolist()    # 886 389

            if len(drugs_feature[0]) > len(targets_feature[0]):
                targets_feature = [lst + [0] * (len(drugs_feature[0]) - len(targets_feature[0])) for lst in targets_feature]
            else:
                drugs_feature = [lst + [0] * (len(targets_feature[0]) - len(drugs_feature[0])) for lst in drugs_feature]
            feature = torch.cat([torch.Tensor(drugs_feature), torch.Tensor(targets_feature)])
        elif args.dataset_name in ["enzyme", "gpcr", "ic", "nr"]:
            dg = pd.read_csv("dataset/YAM/{}/{}_simmat_dg.txt".format(args.dataset_name, args.dataset_name), sep="\t",
                             index_col=0).values.tolist()
            dc = pd.read_csv("dataset/YAM/{}/{}_simmat_dc.txt".format(args.dataset_name, args.dataset_name), sep="\t",
                             index_col=0).values.tolist()
            print(len(dg), len(dc))
            if len(dg) > len(dc):
                dc = [lst + [0] * (len(dg) - len(dc)) for lst in dc]
            else:
                dg = [lst + [0] * (len(dc) - len(dg)) for lst in dg]
            feature = torch.cat([torch.Tensor(dg), torch.Tensor(dc)])

            adj = pd.read_csv("dataset/YAM/{}/{}_admat_dgc.txt".format(args.dataset_name, args.dataset_name), sep="\t",
                              index_col=0).values.tolist()
            edge_index = []
            for i in range(len(adj)):
                for j in range(len(adj[0])):
                    if adj[i][j] == 1:
                        edge_index.append([i, j + len(adj)])
            print(len(edge_index))
            edge_index = torch.LongTensor(edge_index).T

        return Data(x=feature, edge_index=edge_index)

    def len(self) -> int:
        return len(self.data)

    def get(self, idx):
        return self.data


class HashDataset(Dataset):
    def __init__(self, args, data, pos_edges, neg_edges):
        self.elph_hashes = Hashes(args)  # object for hash and subgraph feature operations
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.max_hash_hops = args.hops
        self.subgraph_features = None
        super(HashDataset, self).__init__()
        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)
        self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        self.edge_index = data.edge_index
        self.x = self._preprocess_node_features(data, self.edge_index, self.edge_weight)
        self._preprocess_subgraph_features(data.num_nodes)

    def _generate_sign_features(self, data, edge_index, edge_weight):
        num_nodes = data.x.size(0)
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight.float(), num_nodes)
        xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        return xs

    def _preprocess_node_features(self, data, edge_index, edge_weight):
        x = self._generate_sign_features(data, edge_index, edge_weight)
        return x

    def _preprocess_subgraph_features(self, num_nodes):
        hashes, cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
        self.subgraph_features = self.elph_hashes.get_subgraph_features(self.links, hashes, cards)
        if self.subgraph_features is not None:
            self.subgraph_features[self.subgraph_features < 0] = 0
        if self.subgraph_features is not None:
            if self.max_hash_hops > 1:
                self.subgraph_features[:, [4, 5]] = 0
            if self.max_hash_hops == 3:
                self.subgraph_features[:, [11, 12]] = 0

    def len(self):
        return len(self.links)

    def get(self, idx):
        pass


class MinhashPropagation(MessagePassing):
    def __init__(self):
        super(MinhashPropagation, self).__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=-x)
        return -out


class HllPropagation(MessagePassing):
    def __init__(self):
        super(HllPropagation, self).__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out


class Hashes(object):
    def __init__(self, args):
        self.max_hops = args.hops
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_minhash = np.uint64((1 << 32) - 1)
        self._minhash_range = (1 << 32)
        self.num_perm = 128
        self.minhash_prop = MinhashPropagation()
        self.p = 8
        self.m = 1 << self.p
        tmp = HyperLogLogPlusPlus(p=self.p)
        self.alpha = tmp.alpha
        self.max_rank = tmp.max_rank
        self.hll_size = len(tmp.reg)
        self.hll_threshold = hyperloglog_const._thresholds[self.p - 4]
        self.bias_vector = torch.tensor(hyperloglog_const._bias[self.p - 4], dtype=torch.float)
        self.estimate_vector = torch.tensor(hyperloglog_const._raw_estimate[self.p - 4], dtype=torch.float)
        self.hll_prop = HllPropagation()

    def _np_bit_length(self, bits):
        return np.ceil(np.log2(bits + 1)).astype(int)

    def _get_hll_rank(self, bits):
        bit_length = self._np_bit_length(bits)
        rank = self.max_rank - bit_length + 1
        return rank

    def _init_permutations(self, num_perm):
        gen = np.random.RandomState(1)
        return np.array([
            (gen.randint(1, self._mersenne_prime, dtype=np.uint64),
             gen.randint(0, self._mersenne_prime, dtype=np.uint64)) for _ in range(num_perm)
        ], dtype=np.uint64).T

    def initialise_minhash(self, n_nodes):
        init_hv = np.ones((n_nodes, self.num_perm), dtype=np.int64) * self._max_minhash
        a, b = self._init_permutations(self.num_perm)
        hv = hash_array(np.arange(1, n_nodes + 1))
        phv = np.bitwise_and((a * np.expand_dims(hv, 1) + b) % self._mersenne_prime, self._max_minhash)
        hv = np.minimum(phv, init_hv)
        return torch.tensor(hv, dtype=torch.int64)

    def initialise_hll(self, n_nodes):
        regs = np.zeros((n_nodes, self.m), dtype=np.int8)
        hv = hash_array(np.arange(1, n_nodes + 1))
        reg_index = hv & (self.m - 1)
        bits = hv >> self.p
        ranks = self._get_hll_rank(bits)
        regs[np.arange(n_nodes), reg_index] = np.maximum(regs[np.arange(n_nodes), reg_index], ranks)
        return torch.tensor(regs, dtype=torch.int8)

    def build_hash_tables(self, num_nodes, edge_index):
        hash_edge_index, _ = add_self_loops(edge_index)
        cards = torch.zeros((num_nodes, self.max_hops))
        node_hashings_table = {}
        for k in range(self.max_hops + 1):
            node_hashings_table[k] = {'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8),
                                      'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64)}
            if k == 0:
                node_hashings_table[k]['minhash'] = self.initialise_minhash(num_nodes)
                node_hashings_table[k]['hll'] = self.initialise_hll(num_nodes)
            else:
                node_hashings_table[k]['hll'] = self.hll_prop(node_hashings_table[k - 1]['hll'], hash_edge_index)
                node_hashings_table[k]['minhash'] = self.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                      hash_edge_index)
                cards[:, k - 1] = self.hll_count(node_hashings_table[k]['hll'])
        return node_hashings_table, cards

    def _get_intersections(self, edge_list, hash_table):
        intersections = {}
        for k1 in range(1, self.max_hops + 1):
            for k2 in range(1, self.max_hops + 1):
                src_hll = hash_table[k1]['hll'][edge_list[:, 0]]
                src_minhash = hash_table[k1]['minhash'][edge_list[:, 0]]
                dst_hll = hash_table[k2]['hll'][edge_list[:, 1]]
                dst_minhash = hash_table[k2]['minhash'][edge_list[:, 1]]
                jaccard = self.jaccard(src_minhash, dst_minhash)
                unions = self._hll_merge(src_hll, dst_hll)
                union_size = self.hll_count(unions)
                intersection = jaccard * union_size
                intersections[(k1, k2)] = intersection
        return intersections

    def _linearcounting(self, num_zero):
        return self.m * torch.log(self.m / num_zero)

    def _estimate_bias(self, e):
        nearest_neighbors = torch.argsort((e.unsqueeze(-1) - self.estimate_vector.to(e.device)) ** 2)[:, :6]
        return torch.mean(self.bias_vector.to(e.device)[nearest_neighbors], dim=1)

    def _refine_hll_count_estimate(self, estimate):
        idx = estimate <= 5 * self.m
        estimate_bias = self._estimate_bias(estimate)
        estimate[idx] = estimate[idx] - estimate_bias[idx]
        return estimate

    def hll_count(self, regs):
        if regs.dim() == 1:
            regs = regs.unsqueeze(dim=0)
        retval = torch.ones(regs.shape[0], device=regs.device) * self.hll_threshold + 1
        num_zero = self.m - torch.count_nonzero(regs, dim=1)
        idx = num_zero > 0
        lc = self._linearcounting(num_zero[idx])
        retval[idx] = lc
        estimate_indices = retval > self.hll_threshold
        e = (self.alpha * self.m ** 2) / torch.sum(2.0 ** (-regs[estimate_indices]), dim=1)
        e = self._refine_hll_count_estimate(e)
        retval[estimate_indices] = e
        return retval

    def _hll_merge(self, src, dst):
        return torch.maximum(src, dst)

    def jaccard(self, src, dst):
        return torch.count_nonzero(src == dst, dim=-1) / self.num_perm

    def get_subgraph_features(self, links, hash_table, cards, batch_size=11000000):
        if links.dim() == 1:
            links = links.unsqueeze(0)
        link_loader = DataLoader(range(links.size(0)), batch_size, shuffle=False, num_workers=0)
        all_features = []
        for batch in link_loader:
            intersections = self._get_intersections(links[batch], hash_table)
            cards1, cards2 = cards.to(links.device)[links[batch, 0]], cards.to(links.device)[links[batch, 1]]
            features = torch.zeros((len(batch), self.max_hops * (self.max_hops + 2)), dtype=float, device=links.device)
            features[:, 0] = intersections[(1, 1)]
            if self.max_hops == 1:
                features[:, 1] = cards2[:, 0] - features[:, 0]
                features[:, 2] = cards1[:, 0] - features[:, 0]
            elif self.max_hops == 2:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = cards2[:, 0] - torch.sum(features[:, 0:2], dim=1)  # (0, 1)
                features[:, 5] = cards1[:, 0] - features[:, 0] - features[:, 2]  # (1, 0)
                features[:, 6] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1)  # (0, 2)
                features[:, 7] = cards1[:, 1] - features[:, 0] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]  # (2, 0)
            elif self.max_hops == 3:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = intersections[(3, 1)] - features[:, 0] - features[:, 1]  # (3,1)
                features[:, 5] = intersections[(1, 3)] - features[:, 0] - features[:, 2]  # (1, 3)
                features[:, 6] = intersections[(3, 2)] - torch.sum(features[:, 0:4], dim=1) - features[:, 4]  # (3,2)
                features[:, 7] = intersections[(2, 3)] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]  # (2,3)
                features[:, 8] = intersections[(3, 3)] - torch.sum(features[:, 0:8], dim=1)  # (3,3)
                features[:, 9] = cards2[:, 0] - features[:, 0] - features[:, 1] - features[:, 4]  # (0, 1)
                features[:, 10] = cards1[:, 0] - features[:, 0] - features[:, 2] - features[:, 5]  # (1, 0)
                features[:, 11] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 6] - features[:, 9]  # (0, 2)
                features[:, 12] = cards1[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 7] - features[:, 10]  # (2, 0)
                features[:, 13] = cards2[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 9] - features[:, 11]  # (0, 3)
                features[:, 14] = cards1[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 10] - features[:, 12]  # (3, 0)
            else:
                raise NotImplementedError("Only 1, 2 and 3 hop hashes are implemented")
            features[features < 0] = 0
            all_features.append(features)
        features = torch.cat(all_features, dim=0)
        return features
