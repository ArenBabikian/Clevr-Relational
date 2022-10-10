import json
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import h5py
import random

# TODO get rid of these
neuron_dir_path = "_results/4000images/neurons"
feature_dir_path = "_results/4000images/node2feature"
iepvqa_h5_path = "_data/iepvqa/val_features_0_3000.h5"
layer_id = 2 # 1 for test set

class SceneGraphDatasubset(Dataset):
    def __init__(self, full_dataset, min_id=0, max_id=None):
        if max_id == None:
            max_id = len(full_dataset)

        self.min_id = min_id

        self.scenes = []
        for i in range(min_id, max_id):
            data_obj = full_dataset.get(i)
            self.scenes.append((data_obj.x.detach(), data_obj.edge_index.detach(), data_obj.edge_attr.detach(), data_obj.y.detach()))
            
        # TODO we might have a bug in the adge_attr list. shape shouldbe [90, 4], and not [180, 4]

        # TODO do something in the InMemoryDataset such that the data is stored in a generic way, 
        # and the edge and object transformations are happening in here (SceneGraphDatasubset).
        # Get inspired by Percy's code


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index) -> T_co:
        #TODO improve, send out the Data object directy, then in the rgcn_model.py file, fix the "batch" object.
        # basically the output of this function is the value of batch
        
        #### TODO MAYBE we don't even need to detach the data objects saved in memory? 
        # Because here we are basiclly repacking them as tensors.
        # but this could be an issue if, for example, we wanto pack multiple objects together. In that case, the below code would be required.
        # I am just not sure about the semantics of __getitem__


        # return self.full_dataset.get(index), self.min_id + index
        
        x, edge_list, edge_attr, y = self.scenes[index]
        x_t = torch.FloatTensor(x)
        edge_list_t = torch.LongTensor(edge_list)
        edge_features_t = torch.FloatTensor(edge_attr)
        y_t = torch.FloatTensor(y)
        return Data(x_t, edge_list_t, edge_features_t, y_t), self.min_id + index

class SceneGraphDataset(InMemoryDataset):
    # TODO Update for IEPVQA
    # This is a dataset collecting the neuron values OR features data related to ALL images
    # it basically reads and collects from the raw data saved for each model in the `_results` folder
    # the entire dataset is saved as a local .pt file
    def __init__(self, args):
        self.scenes_path = args.obj_ann_path
        self.schema_path = args.schema_path
        self.data_to_process = args.intermediate_gt
        self.dataset_name = args.dataset_name

        super().__init__(f'{args.dataset_dir}/{self.dataset_name}')
        self.data, self.slices, = torch.load(self.processed_paths[0]) # this is specific to "InMemoryDatasets"        

    def get_reverse_attr_map(self, attribute_map):
        attr_values = [v for values in attribute_map.values() for v in values]
        return {v: i for i, v in enumerate(attr_values)}

    @property
    def processed_file_names(self):
        return [f'data-{self.data_to_process}.pt']

    def process(self):

        # READ SCHEMA
        with open(self.schema_path) as f:
            schema = json.load(f)

        attr2id = self.get_reverse_attr_map(schema['attributes'])
        num_attr = len(attr2id)
        attr_keys = list(schema['attributes'].keys())

        rel2id = {v: i for i, v in enumerate(schema['relations'])}
        num_rels = len(rel2id)

        # READ THE SCENES DATA
        data_list = []
        with open(self.scenes_path, 'r') as f:
            scenes = json.load(f)['scenes']

        # ITERATE THROUGH THE SCENES

        # Create random ordering. Currently only working for IEP:
        # TODO introduce a deterministic seed, for reproductibility 
        random_sequence = list(range(len(scene)))
        random.shuffle(random_sequence)
                    
        for scene in tqdm(scenes):
            ind = scene['image_index']
            objects = scene['objects']
            relations = scene['relationships']

            # CREATE NODES - One-hot encoding
            x_list = []
            for obj in objects:
                obj_feat = [0 for _ in range(num_attr)]
                for attr in attr_keys:
                    obj_feat[attr2id[obj[attr]]] = 1
                x_list.append(obj_feat)

            x = torch.tensor(x_list, dtype=torch.float)

            # CREATE EDGES -  One-hot encoding

            # TODO IDEA 1:
            # maybe add "ghost" edges to make the graph fully conected?
            # as such, the length of y would equal the number of edges

            # TODO IDEA 2:
            # so there are 256 neuron values to learn for each pair of nodes.
            # What we can do is we can have 1 graph for each pair, and we can have new edge types that select the focussed pair, then we would have a list of 256 items as the y.
            # of course, the rest of the graph will be the same (without ghost edges)
            # as such, the size of y will be constant for every model
            # actually, this also will help in testing, cuz we will be able to know the output 256 neuron values actually correspod to which pair of objects, instead of having to try to guess, and running into comparison issues if the NN does not predictthecorrect number of 256-neuron lists

            # TODO MUST DO 3:
            # possibly change the encoding of edges according to the type of encoder (NN) we want to use, possibly given as input
            edge_index_src = []
            edge_index_tgt = []
            edge_attr_list = []

            for rel_type, relations_for_type in relations.items():
                for i_tgt, source_inds in enumerate(relations_for_type):
                    for i_src in source_inds:
                        # add edge
                        edge_index_src.append(i_src)
                        edge_index_tgt.append(i_tgt)

                        # create edge attribute (type)
                        # one-hot encoding
                        edge_feat = [0 for _ in range(num_rels)]
                        edge_feat[rel2id[rel_type]] = 1
                        edge_attr_list.append(edge_feat)
            
            edge_index = torch.tensor([edge_index_src, edge_index_tgt], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

            # GET EXPECTED OUTCOME (feature or neuron values)
            
            if self.dataset_name == "SCENES":
                if self.data_to_process == "relations":
                    neuron_path = f'{neuron_dir_path}/{ind}.pt'
                    neuron_data = torch.load(neuron_path)
                    y = neuron_data['layers'][layer_id]
                elif self.data_to_process == "features":
                    feature_path = f'{feature_dir_path}/{ind}.pt'
                    y = torch.load(feature_path)
                elif self.data_to_process.startswith("random"):
                    # TODO introduce a deterministic seed, for reproductibility 
                    feature_path = f'{feature_dir_path}/{ind}.pt'
                    features = torch.load(feature_path)
                    lb = torch.min(features)
                    ub = torch.max(features)
                    y = (ub-lb) * torch.rand(np.shape(features)) + lb
                else:
                    print(" Data-to-process is not specified...")
                    exit(1)
            elif self.dataset_name == 'IEPVQA':
                if self.data_to_process == "features":
                    with h5py.File(iepvqa_h5_path, 'r') as f :
                        y = torch.FloatTensor(f['features'][ind])
                        y = y.unsqueeze(0)        
                # TODO adjust below
                # elif self.data_to_process.startswith("randommore"):
                #     with h5py.File(iepvqa_h5_path, 'r') as f :
                #         # TODO introduce a deterministic seed, for reproductibility 
                #         real_feat = f['features'][ind]
                #         real_shape = real_feat.shape # 1024-14-14
                #         # print(real_shape)
                #         all_feats = f['features'][self.min_id:self.max_id]
                #         all_shape = all_feats.shape # 3000-1024-14-14
                #         # print(all_shape)
                #         real_count = real_feat.size
                #         # print(real_count)
                #         y = torch.empty(real_shape)

                #         # # Number of dimensions hard-coded
                #         # for i in range(real_shape[0]):
                #         #     for j in range(real_shape[1]):
                #         #         for k in range(real_shape[2]):
                #         #             # value = all_feats
                #         #             # for s in all_shape:
                #         #             rand_i = random.randint(0, all_shape[0]-1)
                #         #             rand_j = random.randint(0, all_shape[1]-1)
                #         #             rand_k = random.randint(0, all_shape[2]-1)
                #         #             rand_l = random.randint(0, all_shape[3]-1)
                #         #                 # print(value.shape)
                #         #                 # value = value[rand_i]
                #         #             # y[i] = value.item()
                #         #             y[i][j][k] = all_feats[rand_i][rand_j][rand_k][rand_l].item()
                        
                #         # Number of dimensions hard-coded
                #         for i in range(real_shape[0]):
                #             # value = all_feats
                #             # for s in all_shape:
                #             rand_i = random.randint(0, all_shape[0]-1)
                #             rand_j = random.randint(0, all_shape[1]-1)
                #                 # print(value.shape)
                #                 # value = value[rand_i]
                #             # y[i] = value.item()
                #             y[i] = torch.from_numpy(all_feats[rand_i][rand_j])


                #         # y = morerandom_sequence[start_id:start_id+real_numel]
                #         # torch.reshape(y, real_shape)
                #         # y.reshape(real_size)
                #         y = y.unsqueeze(0)
                #         # start_id = start_id + real_numel
                #         # print(y.size())
                #         # exit()
                elif self.data_to_process.startswith("random"):
                    real_feat = torch.FloatTensor(f['features'][ind])
                    with h5py.File(iepvqa_h5_path, 'r') as f :
                        real_feat = torch.FloatTensor(f['features'][ind])
                        y = torch.FloatTensor(f['features'][random_sequence[ind]])
                        assert y.size() == real_feat.size()
                        y = y.unsqueeze(0) 
                    # with h5py.File(iepvqa_h5_path, 'r') as f :
                    #     # TODO introduce a deterministic seed, for reproductibility 
                    #     real_feat = torch.FloatTensor(f['features'][ind])
                    #     real_size = real_feat.size()
                    #     real_numel = torch.numel(real_feat)
                    #     print(real_numel)
                    #     y = morerandom_sequence[start_id:start_id+real_numel]
                    #     print(len(y))
                    #     y.reshape(real_size)
                    #     y = y.unsqueeze(0)
                    #     start_id = start_id + real_numel
                    #     print(y.size())
                    #     exit()
                else:
                    print(" Data-to-process is not specified...")
                    exit(1)
            else:
                print("Dataset root directory not")
                exit(1)

            #CREATE torch_geometric.Data.Data OBJECT TO SAVE
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(graph)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
