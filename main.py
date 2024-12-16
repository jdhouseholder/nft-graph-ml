import pickle
import os
from collections import defaultdict
import csv
import time
import argparse

import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear, PReLU, Sequential, ModuleList
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, SAGEConv, to_hetero

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-id",
    type=str
)
parser.add_argument(
    "--lr",
    type=float,
)
parser.add_argument(
    "--conv",
    type=str,
    choices=["SAGEConv", "GraphConv"]
)
parser.add_argument(
    "--decoder",
    type=str,
    choices=["LinearDecoder", "DotProductDecoder"]
)
parser.add_argument(
    "--hidden-layers",
    type=int
)
parser.add_argument(
    "--batch-size",
    type=int,
    choices=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
)
parser.add_argument(
    "--epochs",
    type=int,
    nargs='?',
    default=25
)
parser.add_argument("data_path")
args = parser.parse_args()

experiment_id = f"conv={args.conv}-decoder={args.decoder}-hidden_layers={args.hidden_layers}-batch_size={args.batch_size}-lr={args.lr}"
print(f"Running run_id={args.run_id} experiment_id={experiment_id}")

# Setup some constants.

EVENT_ROOT = f"./events"
if args.run_id:
    EVENT_ROOT += "/" + args.run_id

DATA_PATH = args.data_path

COMPILE = False

EPOCHS = args.epochs

LR = args.lr
WEIGHT_DECAY = 0.0001

BATCH_SIZE = 2048

HIDDEN_LAYERS = 4

HIDDEN_CHANNELS = 64
NEG_SAMPLING_RATIO = 2.0

VERBOSE = True
NON_BLOCKING = True


KEY = ("user", "rx", "token")

CONV = None
if args.conv == "SAGEConv":
    CONV = SAGEConv
elif args.conv == "GraphConv":
    CONV = GraphConv
else:
    raise ValueError(f"invalid conv got {args.conv}")

# Define our model and components.

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_layers, conv):
        super().__init__()
        self.conv_in = conv(hidden_channels, hidden_channels)
        self.activ_in = PReLU()

        self.hidden_layers = hidden_layers

        self._convs = ModuleList()
        self._activations = ModuleList()

        for _ in range(hidden_layers):
            self._convs.append(conv(hidden_channels, hidden_channels))
            self._activations.append(PReLU())

        self.conv_out = conv(hidden_channels, hidden_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.activ_in(self.conv_in(x, edge_index))
        for i in range(self.hidden_layers):
            conv = self._convs[i]
            activ = self._activations[i]

            x = x + activ(conv(x, edge_index))

        x = x + self.conv_out(x, edge_index)
        return x


class LinearDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = Linear(2 * hidden_channels, 1)


    def forward(
        self,
        x_from: Tensor,
        x_to: Tensor,
        edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_from = x_from[edge_label_index[0]]
        edge_feat_to = x_to[edge_label_index[1]]

        a = torch.cat([edge_feat_from, edge_feat_to], dim=-1)
        out = self.lin(a)
        out = out.squeeze()

        return out


class DotProductDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

    def forward(
        self,
        x_from: Tensor,
        x_to: Tensor,
        edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_from = x_from[edge_label_index[0]]
        edge_feat_to = x_to[edge_label_index[1]]
        return (edge_feat_from * edge_feat_to).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_layers, conv, decoder, data):
        super().__init__()
        self.from_emb = torch.nn.Embedding(
            data["user"].num_nodes,
            hidden_channels
        )
        self.to_emb = torch.nn.Embedding(
            data["token"].num_nodes,
            hidden_channels
        )
        self.hetero_gnn = to_hetero(
            GNN(hidden_channels, hidden_layers, conv),
            metadata=data.metadata()
        )
        self.classifier = decoder(hidden_channels)


    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.from_emb(data["user"].node_id),
          "token": self.to_emb(data["token"].node_id),
        }
        x_dict = self.hetero_gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["token"],
            data[KEY].edge_label_index,
        )
        return pred


class EventWriter:
    def __init__(self, root):
        try:
            os.makedirs(root)
        except OSError:
            pass

        self.root = root
        self.events = dict()

    def add(self, key, event):
        f = None
        w = None

        if key not in self.events:
            f = open(f"{self.root}/{key}.csv", "w+")
            self.events[key] = f
            w = csv.DictWriter(f, event.keys(), delimiter=",")
            w.writeheader()
        else:
            f = self.events[key]
            w = csv.DictWriter(f, event.keys(), delimiter=",")

        w.writerow(event)
        f.flush()


# After decoders are defined.
DECODER = None
if args.decoder == "DotProductDecoder":
    DECODER = DotProductDecoder
elif args.decoder == "LinearDecoder":
    DECODER = LinearDecoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(DATA_PATH, "rb") as f:
    tx_data = pickle.load(f)

# Build HeteroData.
data = HeteroData()

data["user"].node_id = torch.arange(tx_data["n_from"])
data["token"].node_id = torch.arange(tx_data["n_to"])

data[KEY].edge_index = torch.tensor(
    tx_data["rx_edges"],
    dtype=torch.long
).t().contiguous()
data[KEY].time = torch.tensor(
    tx_data["rx_block_numbers"],
    dtype=torch.long
).t().contiguous()
data[KEY].edge_label = torch.ones(len(tx_data["rx_block_numbers"]))

# Add a reverse relation for message passing:
data = T.ToUndirected()(data)
# Remove "reverse" label.
del data["token", "rev_rx", "user"].edge_label

# Perform a 80/20 temporal link-level split:
perm = torch.argsort(data[KEY].time)
train_idx = perm[:int(0.8 * perm.size(0))]
val_idx = perm[int(0.8 * perm.size(0)):]


def new_loader(data, idx):
    edge_index = data[KEY].edge_index
    edge_label_index = (KEY, edge_index[:, idx])
    edge_label = data[KEY].edge_label[idx]
    edge_label_time = data[KEY].time[idx] - 1

    return LinkNeighborLoader(
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        edge_label_time=edge_label_time,
        shuffle=True,
        data=data,
        num_neighbors=[5] * (HIDDEN_LAYERS + 2),
        batch_size=BATCH_SIZE,
        time_attr='time',
        temporal_strategy='last',
        num_workers=4,
        neg_sampling="binary",
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        persistent_workers=True,
        pin_memory=True,
    )


train_loader = new_loader(data, train_idx)
val_loader = new_loader(data, val_idx)

event_writer = EventWriter(EVENT_ROOT)

model = Model(
    hidden_channels=HIDDEN_CHANNELS,
    hidden_layers=HIDDEN_LAYERS,
    conv=CONV,
    decoder=DECODER,
    data=data
).to(device)

if COMPILE:
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)

# Turn weight decay off for prelus.
params = []
params_no_weight_decay = []
for name, param in model.named_parameters():
    if 'prelu' in name:
        no_weight_decay.append(param)
    else:
        params.append(param)

optimizer = torch.optim.Adam([
    {"params": params, 'weight_decay': WEIGHT_DECAY},
    {"params": params_no_weight_decay, 'weight_decay': 0.0},
], lr=LR)


def train(device, model, key, train_loader):
    total_loss = 0
    total_examples = 0

    preds = []
    ground_truths = []

    for sampled_data in tqdm.tqdm(train_loader, mininterval=5, miniters=3):
        optimizer.zero_grad()

        sampled_data = sampled_data.to(device, non_blocking=NON_BLOCKING)

        pred = model(sampled_data)

        ground_truth = sampled_data[KEY].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

        loss.backward()

        optimizer.step()

        preds.extend(pred.tolist())
        ground_truths.extend(ground_truth.tolist())

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    return total_loss, total_examples, preds, ground_truths


@torch.no_grad()
def validate(device, model, key, val_loader):
    total_loss = 0
    total_examples = 0
    preds = []
    ground_truths = []

    for sampled_data in tqdm.tqdm(val_loader):
        sampled_data.to(device, non_blocking=NON_BLOCKING)

        pred = model(sampled_data)
        ground_truth = sampled_data[key].edge_label

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

        preds.append(pred)
        ground_truths.append(ground_truth)

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    return total_loss, total_examples, pred, ground_truth


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Put all of the metrics into m for logging.
def compute_and_add_metrics(m, prefix, pred, ground_truth, verbose=False):
    pred = sigmoid(np.asarray(pred))
    ground_truth = np.asarray(ground_truth)

    roc_auc = roc_auc_score(ground_truth, pred)
    avg_precision = average_precision_score(ground_truth, pred)

    m[f"{prefix}:roc_auc"] = roc_auc
    m[f"{prefix}:avg_precision"] = avg_precision


    for t in [0.5, 0.75, 0.9]:
        pred_t = pred > t
        accuracy = np.mean(pred_t == ground_truth)
        precision = precision_score(ground_truth, pred_t)
        recall = recall_score(ground_truth, pred_t)

        t_n, f_p, f_n, t_p = confusion_matrix(ground_truth, pred_t).ravel()

        m[f"{prefix}:accuracy:{t}"] = accuracy
        m[f"{prefix}:precision:{t}"] = precision
        m[f"{prefix}:recall:{t}"] = recall
        m[f"{prefix}:t_p:{t}"] = t_p
        m[f"{prefix}:f_p:{t}"] = f_p
        m[f"{prefix}:t_n:{t}"] = t_n
        m[f"{prefix}:f_n:{t}"] = f_n

        if verbose:
            print()
            print(f"prefix={prefix}")
            print(f"Metrics: t={t}, n={len(ground_truth)}, accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}")
            print(f"Confusion: t_p={t_p:.4f}, f_p={f_p:.4f}, t_n={t_n:.4f}, f_n={f_n:.4f}")
            
    if verbose:
        print()
        print(f"roc_auc={roc_auc:.4f}")
        print(f"avg_precision={avg_precision:.4f}")

    return m

for epoch in range(EPOCHS):
    m = {}

    epoch_start_time = time.time()

    total_loss, total_examples, preds, ground_truths = train(device, model, KEY, train_loader)
    train_loss = total_loss / total_examples
    m["train_loss"] = train_loss

    epoch_time_elapsed = time.time() - epoch_start_time
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Duration: {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    compute_and_add_metrics(
        m,
        "train",
        preds,
        ground_truths,
        verbose=False
    )

    total_loss, total_examples, preds, ground_truths = validate(device, model, KEY, val_loader)
    val_loss = total_loss / total_examples
    m["val_loss"] = val_loss

    compute_and_add_metrics(
        m,
        "val",
        preds,
        ground_truths,
        verbose=VERBOSE
    )

    event_writer.add(f"{experiment_id}-training", m)
