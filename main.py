import torch
import argparse
import numpy as np
from model import Model
from utile import set_seed, get_data, get_loaders, train_func
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help="random seed of dataset and model")
parser.add_argument('--dataset_name', type=str, default='enzyme',
                    choices=['BindingDB', 'davis', 'enzyme', 'gpcr', 'ic', 'nr'])
parser.add_argument('--num_test', type=float, default=0.2, help='ratio of test datasets')
parser.add_argument('--ratio', type=float, default=1, help='ratio of positive samples and negative samples')
parser.add_argument('--task', type=str, default='SP', choices=['SD', 'ST', 'SP'])

parser.add_argument('--dm', type=bool, default=True, help="Whether to use diffusion model")
parser.add_argument('--dm_layers', type=int, default=2, help="The number of layers in the diffusion model")
parser.add_argument('--dm_heads', type=int, default=4, help="The number of heads in the diffusion model")
parser.add_argument('--dm_residua', type=bool, default=True, help="Whether to use residua in the diffusion model")
parser.add_argument('--dm_graph', type=bool, default=True, help="Whether to use graph in the diffusion model")

parser.add_argument('--sf', type=bool, default=True, help="Whether to use subgraph features")
parser.add_argument('--hops', type=int, default=3, help="k-hop subgraph[1,2,3]")
parser.add_argument('--s_dim', type=int, default=64, help="feature dimension of subgraph")

parser.add_argument('--hidden_dim', type=int, default=256)

parser.add_argument('--train_times', type=int, default=10, help='number of training times')
parser.add_argument('--train_epoch', type=int, default=1000, help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of dataset')
args = parser.parse_args()

set_seed(args.seed)
all_result = []
for _ in range(args.train_times):
    dataset, splits = get_data(args)
    train_loader, val_loader, test_loader = get_loaders(args, splits)

    model = Model(args, dataset.num_features).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    result = train_func(args, train_loader, val_loader, test_loader, model, optimizer, loss_fn)
    all_result.append(result)


def print_result(result):
    metrics = ['auc', 'ap', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(float(val[-6:]))
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)


print_result(all_result)

