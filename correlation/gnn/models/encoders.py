import torch
from torch_geometric.nn import Sequential, Linear
from torch_geometric.nn.conv import GATConv, RGCNConv
import torch.nn as nn


class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.encoder = Sequential('x, edge_index, edge_features', [
            (GATConv(-1, 128, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x'),
            (GATConv(-1, 128, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x')])

    def forward(self, x, edge_index, edge_features):
        return self.encoder(x, edge_index, edge_features)

class RGCNEncoder(nn.Module):
    def __init__(self, args, num_features=512):
        super(RGCNEncoder, self).__init__()
        self.encoder = Sequential('x, edge_index, edge_types', [
            (RGCNConv(args.input_channels, num_features, args.num_rels), 'x, edge_index, edge_types -> x'),
            (nn.ReLU(), 'x -> x'),
            # (RGCNConv(num_features, num_features, args.num_rels), 'x, edge_index, edge_types -> x'),
            # (nn.ReLU(), 'x -> x')
        ])

    def forward(self, x, edge_index, edge_features):
        # converting edge feature to edge types
        edge_repeat = edge_features.sum(dim=1).long()
        # repeat the edge based on the number of labels it has
        edge_index = edge_index.T.repeat_interleave(edge_repeat, dim=0).T
        # rever the multi-hot encoding
        edge_types = torch.nonzero(edge_features)[:, 1]

        return self.encoder(x, edge_index, edge_types)


class RGCN2Encoder(nn.Module):
    # TODO do some kind of generalizaion between this and RCGNEncoder class
    def __init__(self, args, num_features=512):
        super(RGCN2Encoder, self).__init__()
        self.encoder = Sequential('x, edge_index, edge_types', [
            (RGCNConv(args.input_channels, num_features, args.num_rels), 'x, edge_index, edge_types -> x'),
            (nn.ReLU(), 'x -> x'),
            (RGCNConv(num_features, num_features, args.num_rels), 'x, edge_index, edge_types -> x'),
            (nn.ReLU(), 'x -> x')
        ])

    def forward(self, x, edge_index, edge_features):
        # converting edge feature to edge types
        edge_repeat = edge_features.sum(dim=1).long()
        # repeat the edge based on the number of labels it has
        edge_index = edge_index.T.repeat_interleave(edge_repeat, dim=0).T
        # rever the multi-hot encoding
        edge_types = torch.nonzero(edge_features)[:, 1]

        return self.encoder(x, edge_index, edge_types)


# IEP FEATURE LEARNING
class GATIEPEncoder(nn.Module):
    # Learns IEP features (size 1024*14*14=200704)
    def __init__(self, args):
        super(GATIEPEncoder, self).__init__()
        self.encoder = Sequential('x, edge_index, edge_features', [
            (GATConv(-1, 256, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x'),
            (GATConv(-1, 256, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x'),
            Linear(1024, 200704)])

    def forward(self, x, edge_index, edge_features):
        return self.encoder(x, edge_index, edge_features)

# POST_STEM IEP FEATURE LEARNING
class GATSTEMEncoder(nn.Module):
    # Learns post-stem IEP features (size 128*14*14=25088)
    def __init__(self, args):
        super(GATSTEMEncoder, self).__init__()
        self.encoder = Sequential('x, edge_index, edge_features', [
            (GATConv(-1, 256, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x'),
            (GATConv(-1, 256, 4, edge_dim=args.edge_dim), 'x, edge_index, edge_features -> x'),
            (nn.ELU(), 'x -> x'),
            Linear(1024, 25088)])

    def forward(self, x, edge_index, edge_features):
        return self.encoder(x, edge_index, edge_features)

# QUESTION ANSWER LEARNING
