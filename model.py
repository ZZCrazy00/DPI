import torch
import torch.nn.functional as F
from transformer_layer import my_Transformer


class Model(torch.nn.Module):
    def __init__(self, args, num_features):
        super(Model, self).__init__()
        self.dm = args.dm
        self.sf = args.sf
        self.hid_dim = args.hidden_dim
        self.sf_dim = args.s_dim

        if self.dm:
            self.encoder_model = my_Transformer(num_features, num_features, num_features, args)

        self.nf_lin1 = torch.nn.Linear(num_features, self.hid_dim)
        self.nf_lin2 = torch.nn.Linear(2 * self.hid_dim, self.hid_dim // 2)
        self.bn2 = torch.nn.BatchNorm1d(self.hid_dim//2)

        if self.sf:
            self.sg_lin = torch.nn.Linear(args.hops * (args.hops + 2), self.sf_dim)
            self.bn1 = torch.nn.BatchNorm1d(self.sf_dim)
            self.output = torch.nn.Linear(self.sf_dim + self.hid_dim//2, 1)
        else:
            self.output = torch.nn.Linear(self.hid_dim//2, 1)

    def feature_forward(self, x):
        x = self.nf_lin1(x)
        x = torch.cat([x[:, 0, :], x[:, 1, :]], 1)
        x = self.nf_lin2(x)
        x = F.dropout(F.relu(self.bn2(x)), p=0.5)
        return x

    def forward(self, data, sample_indices, links, indices):
        if self.dm:
            x = self.encoder_model(data.x.cuda(), data.edge_index.cuda())
        else:
            x = data.x.cuda()
        nf = x[links[indices]].cuda()
        nf = self.feature_forward(nf).to(torch.float)

        if self.sf:
            sf = data.subgraph_features[sample_indices[indices]].cuda().to(torch.float32)
            sf = F.dropout(F.relu(self.bn1(self.sg_lin(sf))), p=0.5)
            x = torch.cat([sf, nf], 1)
        else:
            x = nf
        out = F.sigmoid(self.output(x))
        return out
