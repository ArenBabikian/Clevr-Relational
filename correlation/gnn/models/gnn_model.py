import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GATConv
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torchmetrics import Accuracy
from torch.autograd import Variable
from tqdm import tqdm


import sys
sys.path.append('../clevr-iep')
from iep.data import _dataset_to_tensor
import iep.utils as utils

from correlation.gnn.models.encoders import GATEncoder, GATIEPEncoder, RGCNEncoder, RGCN2Encoder


ENCODER_MAP = {
    'gat': GATEncoder,
    'rgcn': RGCNEncoder,
    'rgcn2': RGCN2Encoder,
    'gatiep': GATIEPEncoder
}

models = {
    '18k': ('program_generator_18k.pt', 'execution_engine_18k.pt'),
    '9k': ('program_generator_9k.pt', 'execution_engine_9k.pt'),
    '700k_strong': ('program_generator_700k.pt', 'execution_engine_700k_strong.pt'),
    'lstm': 'lstm.pt',
    'cnn_lstm': 'cnn_lstm.pt',
    'cnn_lstm_sa': 'cnn_lstm_sa.pt',
    'cnn_lstm_sa_mlp': 'cnn_lstm_sa_mlp.pt'
}

def measureCrossEntropy(t_src, t_tgt):
    # print(t_src.size())
    s_src = torch.nn.Softmax(dim=1)(t_src)
    # print(s_src.size())
    # print(s_src[0][0])
    # t_tgt = torch.LongTensor(list(f_tgt['results']))

    ce = torch.nn.CrossEntropyLoss()(s_src, t_tgt)
    return ce

class SceneConstructionModule(pl.LightningModule):
    def __init__(self, args):
        
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        if self.hparams.source_type == 'IEPVQA-QA':
            models_path = "../clevr-iep/models/CLEVR"
            model_name = models[self.hparams.iep_answer_details[0]]
            
            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            if type(model_name) is tuple:
                program_generator, _ = utils.load_program_generator(f'{models_path}/{model_name[0]}')
                execution_engine, _ = utils.load_execution_engine(f'{models_path}/{model_name[1]}', verbose=False)
                program_generator.type(dtype)
                execution_engine.type(dtype)
                self.model = (program_generator, execution_engine)
                # predicted_programs = []
                # temperature defaulted to 1.0, argmax defaulted to True
                # predicted_programs = program_generator.reinforce_sample(
                #                             q_var,
                #                             temperature=1.0,
                #                             argmax=True)
                # # print("--------")
                # # print(feats[im_i].unsqueeze(0).size())
                # # print(len(predicted_programs))
                # scores = execution_engine(feats[im_i].unsqueeze(0), predicted_programs)
            else:
                model, _ = utils.load_baseline(f'{models_path}/{model_name}')
                model.type(dtype)
                self.model = model
                # scores = model(q_var, feats[im_i].unsqueeze())

            self.criterion = measureCrossEntropy
            self.criterion = torch.nn.MSELoss() # TEMP TODO
            self.accuracy = Accuracy() # TODO we probably want to change this to a more relevant metric for evaluation
        else:
            self.criterion = torch.nn.MSELoss() # because we are comparing float vectors
            self.accuracy = None
                                          
        encoder = ENCODER_MAP[self.hparams.encoder](self.hparams)
        self.net = SceneConstructionModel(encoder, self.hparams.dropout_p, self.hparams.num_rels,
                                          self.hparams.use_sigmoid, self.hparams.decoder)

    def forward(self, batch):
        return self.net(batch.x, batch.edge_index, batch.edge_attr)

    def get_metrics(self, batch):
        if torch.cuda.is_available():
            t = torch.cuda
        else:
            t = torch

        batch_graph, num_obj_seq, scene_ids = batch
        reconstruction = self.forward(batch_graph)

        if self.hparams.source_type == 'IEPVQA-QA':

            # HANDLE LEARNED FEATURE VALUES
            assert torch.sum(num_obj_seq) == len(batch_graph.x)

            # Average out the reconstruction
            # [batch_size * num_nodes, 200704] -> [batch_size, 200704] -> [batch_size, 1024, 14, 14]
            feats = []
            start_id = 0
            for n in num_obj_seq:
                feats.append(torch.mean(reconstruction[start_id:start_id+n.data], 0).reshape(1024, 14, 14))
                start_id = start_id+n.data
            feats = torch.stack(feats)
            # print(feats.size())
            # exit()

            # HANDLE Y
            # [batch_size, num_qs_per_image, 2, question_length]

            score_size = 0
            for score in batch_graph.y[0][0][1]:
                if score != float('-inf'):
                    score_size += 1
            obtained_scores = torch.empty((len(batch_graph.y), len(batch_graph.y[0]), score_size), dtype=torch.float)
            # print(obtained_scores.size())
            gt_scores = torch.empty((len(batch_graph.y), len(batch_graph.y[0])), dtype=torch.long)
            gt_scores_full = torch.empty((len(batch_graph.y), len(batch_graph.y[0]), score_size), dtype=torch.float)
            # print(gt_scores.size())
            # exit()

            # [batch_size, num_qs_pr_image, scores_size]
            # [batch_size * num_qs_pr_image, scores_size]

            for im_i in range(len(batch_graph.y)):
                for q_id, qa_pair in enumerate(batch_graph.y[im_i]):
                    q = qa_pair[0].to(torch.long)
                    q = t.LongTensor(q).view(1, -1)
                    q = q.type(t.FloatTensor).long()
                    q_var = Variable(q)
                    q_var.requires_grad = False

                    a = qa_pair[1]
                    # print(a)
                    a_answer_id = a.argmax()
                    a_full = a[:score_size]
                    # _, a_answer_id = a.data.cpu().max()
                    # print(a_answer_id)
                    # Need to romove the -infs from the end of a, to make it equivalent to scores_size
                    # OR
                    # we need to extract just the solution
                    # exit()

            
                    # dtype = torch.FloatTensor
                    if type(self.model) is tuple:
                        # temperature defaulted to 1.0, argmax defaulted to True
                        
                        program_generator, execution_engine = self.model
                        predicted_programs = program_generator.reinforce_sample(
                                                    q_var,
                                                    temperature=1.0,
                                                    argmax=True)
                        # print("--------")
                        # print(feats[im_i].unsqueeze(0).size())
                        # print(len(predicted_programs))
                        scores = execution_engine(feats[im_i].unsqueeze(0), predicted_programs)
                    else:
                        # model, _ = utils.load_baseline(f'{models_path}/{model_name}')
                        # model.type(dtype)
                        scores = self.model(q_var, feats[im_i].unsqueeze())

            
            
                    # compare scores to answr from q-a pair
                    obtained_scores[im_i][q_id] = scores
                    gt_scores[im_i][q_id] = a_answer_id
                    gt_scores_full[im_i][q_id] = a_full
                    # aggregate
                    # TODO
                
            # exit()
            # print('-----------')

            # FOR MSE
            loss = self.criterion(obtained_scores, gt_scores_full)


            # FOR CROSS-ENTROPY TEMP TODO
            obtained_scores_rs = obtained_scores.reshape((len(batch_graph.y)* len(batch_graph.y[0]), score_size))
            # print(obtained_scores.size())
            gt_scores_rs = gt_scores.reshape((len(batch_graph.y)* len(batch_graph.y[0])))
            # # print(gt_scores.size())
            # # assert torch.all(reshaped[0] == obtained_scores[0][0])
            # # assert torch.all(reshaped_2[0] == gt_scores[0][0])
            # # for i in range(20):
            # #     print(torch.all(reshaped[i] == obtained_scores[1][0]))
            # #     print(torch.all(reshaped_2[i] == gt_scores[1][0]))
            # #     print('------')
            # # exit()
            # loss = self.criterion(obtained_scores_rs, gt_scores_rs) # TODO THIS IS THE LINE FOR CE

            # acc = self.accuracy(torch.add(preds, -m), torch.add(target, -m))
            acc = self.accuracy(obtained_scores_rs.argmax(dim=1), gt_scores_rs)

        elif self.hparams.source_type == 'IEPVQA' or self.hparams.source_type == 'IEPVQA-DIS':
            # Average out the reconstruction
            # [batch_size * num_nodes, 200704] -> [batch_size, 200704]
            assert torch.sum(num_obj_seq) == len(batch_graph.x)

            loss_feat = []
            start_id = 0
            for n in num_obj_seq:
                loss_feat.append(torch.mean(reconstruction[start_id:start_id+n.data], 0))
                start_id = start_id+n.data
            loss_feat = torch.stack(loss_feat)

            # print(loss_feat.size())
            
            # Flatten batch_graph.y
            # [batch_size, 1024, 14, 14] -> [batch_size, 200704]
            loss_gt = torch.flatten(batch_graph.y, start_dim=1)
            # print(batch_graph.y.size())
            # print(loss_gt.shape)
            # exit()
            loss = self.criterion(loss_feat, loss_gt)
            acc = 0.5

        else:
            # SCENES
            loss = self.criterion(reconstruction, batch_graph.y)
            # preds = torch.round(reconstruction).int()
            # target = batch_graph.y.int()
            # m = torch.min(torch.cat((preds, target)))
            acc = 0.5
        # TODO IMPROVE THIS ACCURACY MEASUREMENT
        # acc = self.accuracy(torch.add(preds, -m), torch.add(target, -m))
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
