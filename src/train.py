from model import Model
from data_utils import download_data, get_raw_data, normalize_data, label_anomalies, make_graph_data
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import argparse
import random
import math


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed',  type=int, default=42)
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr_GE', type=float, default=1e-3, help='generator/encoder learning rate')
parser.add_argument('--lr_D', type=float, default=5e-6, help='discriminator learning rate')
parser.add_argument('--cc_weight', type=float, default=0.01, help='cycle consistency coefficient')
parser.add_argument('--dropout_p', type=float, default=0.1, help='dropout rate')
parser.add_argument('--batch_size', type=int, default=32, help='number of samples per forward/backward pass')
parser.add_argument('--verbose',  type=bool, default=False, help='print loss during training')
parser.add_argument('--feature_size', type=int, default=3, help='input feature dimension')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden feature dimension')
parser.add_argument('--latent_size', type=int, default=128, help='latent feature dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--num_nodes', type=int, default=196, help='total number of nodes in the graph')
parser.add_argument('--replicate_latent', type=bool, default=False, help='replicate latent features for every node')
parser.add_argument('--repeats_D', type=int, default=1, help='number of discriminator updates per generator/encoder update')
parser.add_argument('--repeats_GE', type=int, default=2, help='number of generator/encoder updates per discriminator update')
parser.add_argument('--pretrained',  type=bool, default=False, help='use the pretrained model')
args = parser.parse_args(args=[])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '../data'
model_save_path = '../saved_model'

np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

download_data(data_path)
train_df = get_raw_data(data_path + 'nashville_freeway_anomaly.csv')
min_vals = {'speed':math.floor(train_df['speed'].min()), 
            'occ':math.floor(train_df['occ'].min()), 
            'volume':math.floor(train_df['volume'].min())}
max_vals = {'speed':math.ceil(train_df['speed'].max()), 
            'occ':math.ceil(train_df['occ'].max()), 
            'volume':math.ceil(train_df['volume'].max())}

train_df = normalize_data(train_df, min_vals, max_vals)
train_df = label_anomalies(train_df)
train_data = make_graph_data(train_df)
train_loader = DataLoader(train_data, args.batch_size, shuffle=True)

model = Model(args, device)
loss_hist = model.train(train_loader)
model.save_weights(model_save_path)

