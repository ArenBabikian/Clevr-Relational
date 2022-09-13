import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GATConv
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torchmetrics import Accuracy

from correlation.gnn.models.encoders import GATEncoder, RGCNEncoder, RGCN2Encoder


ENCODER_MAP = {
    'gat': GATEncoder,
    'rgcn': RGCNEncoder,
    'rgcn2': RGCN2Encoder
}

class SceneConstructionModule(pl.LightningModule):
    def __init__(self, args):
        
        super().__init__()
        self.save_hyperparameters(args.__dict__)
        self.criterion = torch.nn.MSELoss() # because we are comparing float vectors
                                          
        encoder = ENCODER_MAP[self.hparams.encoder](self.hparams)
        self.net = SceneConstructionModel(encoder, self.hparams.dropout_p, self. hparams.num_rels,
                                          self.hparams.use_sigmoid, self.hparams.decoder)
        self.accuracy = Accuracy() # TODO we probably want to change this to a more relevant metric for evaluation

    def forward(self, batch):
        return self.net(batch.x, batch.edge_index, batch.edge_attr)

    def get_metrics(self, batch):
        x, edge_index, edge_attr, y, scene_ids = batch
        # only support single ids for now
        assert len(scene_ids) == 1
        data = Data(x[0], edge_index[0], edge_attr[0], y[0])
        batch_graph = Batch.from_data_list([data])

        reconstruction = self.forward(batch_graph)
        loss = self.criterion(reconstruction, batch_graph.y)
        # TODO IMPROVE THIS ACCURACY MEASUREMENT
        preds = torch.round(reconstruction).int()
        target = batch_graph.y.int()
        m = torch.min(torch.cat((preds, target)))
        acc = self.accuracy(torch.add(preds, -m), torch.add(target, -m))
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/test', loss)
        self.log('acc/test', accuracy)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [scheduler]

class SceneConstructionModel(nn.Module):
    
    def __init__(self, encoder, dropout_p, num_rels, use_sigmoid, decoder, num_features=512):
    # def __init__(self, num_rels, dropout_p, use_sigmoid, edge_dim, num_features=512):
        super(SceneConstructionModel, self).__init__()
        self.encoder = encoder

        # Original Output layer.
        # NOT used currently for Aren's work
        output_layers = [nn.Dropout(dropout_p),
                        nn.Linear(2 * num_features, 256),
                        nn.Linear(256, num_rels)]
        if use_sigmoid:
            output_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*output_layers)

    def forward(self, x, edge_index, edge_features):
        return self.encoder(x, edge_index, edge_features)
