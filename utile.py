import tqdm
import torch
import random
import numpy as np
import tensorflow as tf
from dataset import MyDataset, HashDataset
from torch.utils.data import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.random.set_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(args):
    dataset = MyDataset(args)
    transform = RandomLinkSplit(is_undirected=True, num_val=0, num_test=args.num_test,
                                add_negative_train_samples=True, neg_sampling_ratio=args.ratio)
    train_data, val_data, test_data = transform(dataset.data)

    if args.task == 'SD':
        print('SD')
        test_drug_set = set(test_data['edge_label_index'][1].tolist())
        train_drug_list = train_data['edge_label_index'][1].tolist()
        edge_label_index = train_data['edge_label_index'].tolist()
        edge_label = train_data['edge_label'].tolist()
        for drug in train_drug_list:
            if drug in test_drug_set:
                index_to_remove = train_drug_list.index(drug)
                edge_label_index = [sl[:index_to_remove] + sl[index_to_remove + 1:] for sl in edge_label_index]
                edge_label = edge_label[:index_to_remove] + edge_label[index_to_remove + 1:]
        train_data['edge_label_index'] = torch.LongTensor(edge_label_index)
        train_data['edge_label'] = torch.Tensor(edge_label)

    elif args.task == 'ST':
        print('ST')
        test_target_set = set(test_data['edge_label_index'][0].tolist())
        train_target_list = train_data['edge_label_index'][0].tolist()
        edge_label_index = train_data['edge_label_index'].tolist()
        edge_label = train_data['edge_label'].tolist()
        for drug in train_target_list:
            if drug in test_target_set:
                index_to_remove = train_target_list.index(drug)
                edge_label_index = [sl[:index_to_remove] + sl[index_to_remove + 1:] for sl in edge_label_index]
                edge_label = edge_label[:index_to_remove] + edge_label[index_to_remove + 1:]
        train_data['edge_label_index'] = torch.LongTensor(edge_label_index)
        train_data['edge_label'] = torch.Tensor(edge_label)
    else:
        print("SP")

    splits = {'train': train_data, 'val': test_data, 'test': test_data}
    return dataset, splits


def get_pos_neg_edges(data):
    pos_edges = data['edge_label_index'][:, data['edge_label'] == 1].t()
    neg_edges = data['edge_label_index'][:, data['edge_label'] == 0].t()
    return pos_edges, neg_edges


def get_hashed_train_val_test_datasets(args, train_data, val_data, test_data):
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    train_dataset = HashDataset(args, train_data, pos_train_edge, neg_train_edge)
    val_dataset = HashDataset(args, val_data, pos_val_edge, neg_val_edge)
    test_dataset = HashDataset(args, test_data, pos_test_edge, neg_test_edge)
    return train_dataset, val_dataset, test_dataset


def get_loaders(args, splits):
    train_data, val_data, test_data = splits['train'], splits['val'], splits['test']
    train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(args, train_data, val_data, test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, test_loader


def train_model(data_loader, model, optimizer, loss_fn):
    model.train()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.randperm(len(labels))[:len(labels)]
    links = data.links[sample_indices]
    labels = labels[sample_indices]
    total_loss = 0
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=512, shuffle=True)):
        optimizer.zero_grad()
        logits = model(data, sample_indices, links, indices)
        loss = loss_fn(logits.view(-1), labels[indices].squeeze(0).cuda().to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss


@torch.no_grad()
def test_model(data_loader, model):
    model.eval()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.arange(0, len(labels))
    links = data.links[sample_indices]
    preds = []
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=8192, shuffle=False)):
        logits = model(data, sample_indices, links, indices, target='test')
        preds.append(logits.view(-1).cpu())
    pred = torch.cat(preds)
    labels = labels[:len(pred)]

    AUC = roc_auc_score(labels, pred)
    AP = average_precision_score(labels, pred)
    temp = torch.tensor(pred)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(labels, temp.cpu())
    return ['AUC:{:.4f}'.format(AUC), 'AP:{:.4f}'.format(AP),
            'acc:{:.4f}'.format(accuracy.item()), 'sen:{:.4f}'.format(sensitivity.item()),
            'pre:{:.4f}'.format(precision.item()), 'spe:{:.4f}'.format(specificity.item()),
            'f1:{:.4f}'.format(F1_score.item()), 'mcc:{:.4f}'.format(mcc.item())]


def train_func(args, train_loader, val_loader, test_loader, model, optimizer, loss_fn):
    early_stop = 0
    best_auc = 0
    output_result = []
    for epoch in tqdm.tqdm(range(args.train_epoch)):
        early_stop += 1
        loss = train_model(train_loader, model, optimizer, loss_fn)
        result = test_model(test_loader, model)
        print(result)
        if float(result[0][4:]) > best_auc:
            early_stop = 0
            best_auc = float(result[0][4:])
            output_result = result
        if early_stop == 10:
            print("Early Stopping", output_result)
            break
    return output_result
