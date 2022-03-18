import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from scene_parse.rel_net.constraints import build_adjacency_matrix, adj_probability
from scene_parse.rel_net.constraints.constraint_loss import get_deduct_constraint, get_anti_symmetry_constraint,\
    get_transitivity_constraint


class SceneBasedRelNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        self.criterion = torch.nn.BCELoss()

        constraints = [get_deduct_constraint([(0, 1), (2, 3)]), get_anti_symmetry_constraint(epsilon=-0.5),
                       get_transitivity_constraint()]
        self.logic_criteria = lambda a: (constraints[0](a) + constraints[1](a) + constraints[2](a)) / torch.numel(a)

        num_rels = self.hparams.num_rels if self.hparams.used_rels is None else len(self.hparams.used_rels)
        self.net = _RelNet(num_rels, self.hparams.dropout_p, self.hparams.use_sigmoid)

    def forward(self, data, sources, targets):
        predictions = self.net(data, sources, targets)
        return predictions

    def get_metrics(self, batch):
        data, sources, targets, labels, _, (nums_obj, nums_edges), _ = batch
        # data, sources, targets = data.squeeze(dim=0), sources.squeeze(dim=0), targets.squeeze(dim=0)
        # labels = labels.squeeze(dim=0)
        n = data.shape[0]
        accuracies = torch.zeros((n, labels.shape[2]))  # accuracies for each label type
        losses = torch.zeros(n)
        constraint_losses = torch.zeros(n)

        # TODO: Find a better way to handle multi scenes in one epoch
        for i in range(n):
            num_obj = nums_obj[i].item()
            num_edges = nums_edges[i].item()
            data_i = data[i][:num_obj]
            sources_i = sources[i][:num_edges].long()
            targets_i = targets[i][:num_edges].long()
            labels_i = labels[i][:num_edges].long()

            predictions = self.forward(data_i, sources_i, targets_i)

            adj_size = int(math.sqrt(predictions.shape[0]))
            adj = build_adjacency_matrix(predictions, adj_size)

            if not self.hparams.use_sigmoid:
                adj = adj_probability(adj)
            adj_gt = build_adjacency_matrix(labels_i, adj_size)

            losses[i] = self.criterion(adj, adj_gt.float())
            constraint_losses[i] = self.logic_criteria(adj)

            adj = torch.round(adj)
            accuracies[i] = (adj == adj_gt).sum(dim=[1, 2]) / predictions.shape[0]

        return losses.mean(), constraint_losses.mean(), accuracies.mean(dim=0)

    def training_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log('constraint_loss/train', constraint_loss)
        self.log(f'acc_overall/train', accuracies.mean())
        self.log('lr', self.scheduler.get_last_lr()[0])
        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/train', acc)

        # labelled = batch[-1].item()

        # return loss if labelled else constraint_loss
        return 0.4 * loss + 0.6 * constraint_loss if self.hparams.include_constraint_loss else loss

    def validation_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log('constraint_loss/val', constraint_loss)
        self.log(f'acc_overall/val', accuracies.mean())

        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/val', acc)
            self.log(f'scene_acc{i}/val', 1 if np.isclose(1, acc.item()) else 0)

    def test_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log(f'acc_overall/test', accuracies.mean())
        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/test', acc)

        return accuracies

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        self.scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [self.scheduler]


class _RelNet(nn.Module):
    def __init__(self, num_rels, dropout_p, use_sigmoid, input_channels=4, num_features=512):
        super(_RelNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.num_features = num_features
        self.feature_extractor = nn.Sequential(*layers)

        # self.rnn = nn.Sequential(nn.Dropout(dropout_p), nn.RNN(num_features, 128, batch_first=False))
        # self.output = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(128, num_rels),
        #                             nn.Sigmoid())
        output_layer = [nn.Dropout(dropout_p), nn.Linear(2 * num_features, 256), nn.Linear(256, num_rels)]
        if use_sigmoid:
            output_layer.append(nn.Sigmoid())
        self.output = nn.Sequential(*output_layer)

        # PART 1: (indiv objects)
        # input = objects + attr
        # output = 512-dim vector for each object

        # PART 2: (object pairs)
        # input = 2 512-dim vector
        # intermediate  = 1024-dim vetor becomes a 256-dim vector
        # output = what is the relation between the 2? (i.e. left, right, behind, in front)

    def _feature_extractor(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, objs, sources, targets):
        # print('inputs')
        # print(objs.shape) #56*4*224*224
        # print(sources) # (56) 1, 1, ... (7 times), 2, 2, ... (7 times)...
        # print(targets) # (56) 1,2,3,4...,0,2,3,4,...,0,1,3,4...

        #CALL first NN
        #gets 512-vector for each object
        features = self._feature_extractor(objs)

        # print('features')
        # print(features.shape) #12*512
        # print(features[sources].shape) #56*512
        # print(features[targets].shape) #56*512

        relations = torch.cat([features[sources], features[targets]], dim=1)
        # print(relations.shape) #56*1024

        ### CHECKING HOW MANY NEURONS ARE ACTIVATED BY EACH ACTOR
        
        #get num actors
        num_objs = (1+math.sqrt(1+4*len(sources)))/2
        num_objs = int(num_objs) if num_objs.is_integer() else exit(1)

        # check how many are activated
        thresh_obj = 0
        for i in range(num_objs):
            obj = features[i]
            num_active = len([j for j in obj if j > thresh_obj])
            print(f'obj{i} = {num_active} /{features.shape[1]}')

        ### GET INTERMEDIATE (256) neuron values
        test_sigmoid = False
        thresh = 0.999 if test_sigmoid else 40
        print(thresh)

        post_dropout = self.output[0](relations) #Dropout
        inter_neuron_vals = self.output[1](post_dropout) # 1024 -> 256
        intermediate_neurons = nn.Sigmoid()(inter_neuron_vals) if test_sigmoid else inter_neuron_vals # (56*1024)
        
        clean_arr = np.full(intermediate_neurons.shape, thresh, dtype=int)
        max_val = 0
        for i in range(len(intermediate_neurons)):
            for j in range(len(intermediate_neurons[i])):
                if intermediate_neurons[i][j] > thresh:
                    clean_arr[i][j] = intermediate_neurons[i][j]
                if intermediate_neurons[i][j] > max_val:
                    max_val = intermediate_neurons[i][j]

        ### SHOW ALL
        import seaborn as sns; sns.set_theme()
        import matplotlib.pyplot as plt
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        ax = sns.heatmap(clean_arr)
        plt.show()

        # check how many are activated
        all_active_ids = set()
        for i in range(len(sources)):
            if i % num_objs == 0:
                print()
            i_src = sources[i]
            i_tgt = targets[i]
            # select with or w/o threshold
            # layer = clean_arr[i]
            layer = intermediate_neurons[i].tolist()

            #Plot CDF
            plt.plot(np.sort(layer), np.arange(256), marker='o')
            plt.show()

            all_active = [j for j in layer if j > thresh]

            # add active ids to set
            for act_id in range(len(layer)):
                if layer[act_id] > thresh:
                    all_active_ids.add(act_id) 

            num_active = len(all_active)
            print(f'[{num_active}][{len(all_active_ids)}]/256', end=', ')

        print('<END>')

        #CALL the second NN on each oriented pair of nodes
        # relation vec, 56*4 (each possible relationshiop is given a probability of being an instance of each edge type)

        return self.output(relations)
