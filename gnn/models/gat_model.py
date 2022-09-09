import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from .encoders import GATEncoder, RGCNEncoder


ENCODER_MAP = {
    'gat': GATEncoder,
    'rgcn': RGCNEncoder
}


class SceneConstructionModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)
        self.criterion = torch.nn.BCELoss()

        encoder = ENCODER_MAP[self.hparams.encoder](self.hparams)
        self.net = SceneConstructionModel(encoder, self.hparams.dropout_p, self. hparams.num_rels,
                                          self.hparams.use_sigmoid, self.hparams.decoder)
        self.accuracy = Accuracy()

    def forward(self, batch):
        return self.net(batch.x, batch.edge_index, batch.edge_attr)

    def get_metrics(self, batch):
        batch_graph, scene_ids = batch

        reconstruction = self.forward(batch_graph)
        loss = self.criterion(reconstruction, batch_graph.y)
        acc = self.accuracy(torch.round(reconstruction), batch_graph.y.int())
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log(f'acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log(f'acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log(f'acc/test', accuracy)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [scheduler]


class SceneConstructionModel(nn.Module):
    def __init__(self, encoder, dropout_p, num_rels, use_sigmoid, decoder, num_features=512):
        super(SceneConstructionModel, self).__init__()
        self.encoder = encoder

        output_layers = [nn.Dropout(dropout_p), nn.Linear(2 * num_features, 256),
                         nn.Linear(256, num_rels)]
        if use_sigmoid:
            output_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*output_layers)
        if 'statedict' in decoder:
            print(decoder)
            self.decoder.load_state_dict(torch.load(decoder))
            for param in self.decoder.parameters():
                param.requires_grad = False
                print(param)
        if decoder == 'random':
            for param in self.decoder.parameters():
                param.requires_grad = False
                print(param)

    def forward(self, x, edge_index, edge_features):
        z = self.encoder(x, edge_index, edge_features)
        rel_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        return self.decoder(rel_features)


