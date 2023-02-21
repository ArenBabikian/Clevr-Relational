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
from correlation.gnn.models.encoders import GATEncoder, GATIEPEncoder, RGCNEncoder, RGCN2Encoder
import sys

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
    # 'lstm': 'lstm.pt',
    # 'cnn_lstm': 'cnn_lstm.pt',
    # 'cnn_lstm_sa': 'cnn_lstm_sa.pt',
    # 'cnn_lstm_sa_mlp': 'cnn_lstm_sa_mlp.pt'
}

def measureCrossEntropy(t_src, t_tgt):
    s_src = torch.nn.Softmax(dim=0)(t_src)
    s_tgt = torch.nn.Softmax(dim=0)(t_tgt)
    ce = torch.nn.CrossEntropyLoss()(s_src, s_tgt)
    return ce

class SceneConstructionModule(pl.LightningModule):
    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters(args.__dict__)

        if args.source_type == 'IEPVQA':
            # TODO for the FUTURE, this also applies to 'IEPVQA-DIS'
            self.criterion = torch.nn.MSELoss()
            self.accuracy = None
        elif args.source_type == 'CLEVR-GNN':
            self.criterion = torch.nn.MSELoss()
            self.accuracy = None
        elif args.source_type == 'IEPVQA-Q':            
            sys.path.append(args.clevr_iep_path)
            import iep.utils as utils
            models_path = f'{args.clevr_iep_path}/models/CLEVR'
            model_name = models[args.decoder]
            
            dtype = torch.cuda.FloatTensor if args.gpu != 0 else torch.FloatTensor
            if type(model_name) is tuple:
                program_generator, _ = utils.load_program_generator(f'{models_path}/{model_name[0]}')
                execution_engine, _ = utils.load_execution_engine(f'{models_path}/{model_name[1]}', verbose=False)
                program_generator.type(dtype)
                execution_engine.type(dtype)
                self.model = (program_generator, execution_engine)
            else:
                model, _ = utils.load_baseline(f'{models_path}/{model_name}')
                model.type(dtype)
                self.model = model

            self.criterion = measureCrossEntropy
            self.accuracy = Accuracy() # TODO we probably want to change this to a more relevant metric for evaluation
                                          
        encoder = ENCODER_MAP[self.hparams.encoder](self.hparams)
        self.net = SceneConstructionModel(encoder, self.hparams.dropout_p, self.hparams.num_rels,
                                          self.hparams.use_sigmoid, self.hparams.decoder)

    def forward(self, batch):
        return self.net(batch.x, batch.edge_index, batch.edge_attr)

    def get_metrics(self, batch):
        t = torch.cuda if torch.cuda.is_available() else torch

        batch_graph, num_obj_seq, scene_ids = batch
        reconstruction = self.forward(batch_graph)

        if self.hparams.source_type == 'IEPVQA':
            # TODO for the FUTURE, this also applies to 'IEPVQA-DIS'

            # RECONSTRUCTION
            # [batch_size * num_nodes, 200704] -> [batch_size, 200704] (floats)
            assert torch.sum(num_obj_seq) == len(batch_graph.x)

            loss_feat = []
            start_id = 0
            for n in num_obj_seq:
                # Average out the reconstruction for each image in batch
                loss_feat.append(torch.mean(reconstruction[start_id:start_id+n.data], 0))
                start_id = start_id+n.data
            # stack all average reconstructions of a same batch
            loss_feat = torch.stack(loss_feat)

            # BATCH
            # [batch_size, 1024, 14, 14] -> [batch_size, 200704] (floats)
            loss_gt = torch.flatten(batch_graph.y, start_dim=1)

            # LOSS
            loss = self.criterion(loss_feat, loss_gt)
            acc = 0.5
        elif self.hparams.source_type == 'CLEVR-GNN':
            loss = self.criterion(reconstruction, batch_graph.y)
            acc = 0.5
        elif self.hparams.source_type == 'IEPVQA-Q':

            # TODO
            # Feed into decoder
            # [batch_size * num_nodes, 200704] -> [batch_size, max_num_nodes, 200704]

            # learning one [1024, 14, 14] feature per node in the input graph
            assert torch.sum(num_obj_seq) == len(batch_graph.x)

            # RECONSTRUCTION
            # Average out the reconstruction (need [batch_size, 1024, 14, 14])
            # [batch_size * num_nodes, 200704] --mean-> [batch_size, 200704] --reshape-> [batch_size, 1024, 14, 14] (floats)
            graph_feats = []
            start_id = 0
            for n in num_obj_seq:
                graph_feats.append(torch.mean(reconstruction[start_id:start_id+n.data], 0).reshape(1024, 14, 14))
                start_id = start_id+n.data
            graph_feats = torch.stack(graph_feats)

            # No score handling for now
            ans_len = self.hparams.answer_length

            # [batch_size, num_qs_pr_image, scores_size]
            # [batch_size * num_qs_pr_image, scores_size]

            tot_loss = torch.zeros((len(batch_graph.y), len(batch_graph.y[0])), dtype=torch.float).cuda()
            succ_rate = torch.zeros((len(batch_graph.y), len(batch_graph.y[0])), dtype=torch.float).cuda()

            # NOTE THIS CALCULATION OF LOSS AND ACC IS WHAT IS TAKING THE MAJORITY OF TIME
            for im_i, im_batch in enumerate(batch_graph.y):
                for q_id, qa_pair in enumerate(im_batch):
                    # features
                    cur_feats = graph_feats[im_i].unsqueeze(0)

                    # Gt Answer: 
                    answer_gt_oh = qa_pair[1][:ans_len].float()

                    # Question
                    q = qa_pair[0].to(torch.long)
                    q = t.LongTensor(q).view(1, -1)
                    q = q.type(t.FloatTensor).long()
                    q_var = Variable(q)
                    q_var.requires_grad = False
                    q_var.unsqueeze(0)

                    # Run decoder on q_var
                    if type(self.model) is tuple:                        
                        program_generator, execution_engine = self.model
                        predicted_programs = program_generator.reinforce_sample(
                                                    q_var,
                                                    temperature=1.0,
                                                    argmax=True)
                        scores = execution_engine(cur_feats, predicted_programs)
                    else:
                        scores = self.model(q_var, cur_feats)

                    # Learned Amswer
                    _, answer_id = scores.data.cpu().max(dim=1)
                    answer_le_oh = torch.zeros(ans_len, dtype=torch.float).cuda()
                    answer_le_oh[int(answer_id.item())] = 1
                    answer_le_oh.requires_grad = True
                    # print('CHANGED')

                    # TODO TRY WITH SCORES
                    # answer_le_oh = scores

                    # Loss
                    # (inputs must be float tensors)
                    cur_loss = self.criterion(answer_le_oh, answer_gt_oh)
                    # self.criterion(scores[0], answer_gt_oh)
                    tot_loss[im_i][q_id] = cur_loss

                    # Accuracy
                    isCorrect = 1 if torch.all(answer_gt_oh == answer_le_oh).item() else 0
                    succ_rate[im_i][q_id] = isCorrect
                
            loss = torch.mean(tot_loss)
            acc = torch.mean(succ_rate)
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
