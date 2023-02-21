import json
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import h5py
import random
import sys
import os
import numpy as np

class SceneGraphDatasubset(Dataset):
    def __init__(self, full_dataset, target_type, min_id=0, max_id=None):
        max_id = len(full_dataset) if max_id == None else max_id
        # print(len(full_dataset))
        self.min_id = min_id
        self.scenes = []

        # Random handling
        rand_no_rep_seq = list(range(len(full_dataset)))
        random.shuffle(rand_no_rep_seq)
        for i in range(min_id, max_id):

            # INPUT
            x = full_dataset.x[i]
            edge_attr = full_dataset.edge_attr[i]
            edge_index = full_dataset.edge_index[i]

            # OUTPUT
            # TODO currently, only 'features' is handled by "CLEVR-GNN" 
            # TODO adjust randomization of y for the IEPVQA-Q case. Question stays the same, answer is selected from another question
            # handle randomization if requred
            if target_type == "features":
                y = full_dataset.y[i]
            elif target_type == "randomwithreplacement":
                random_i = random.randint(0, len(full_dataset.y)-1)
                y = full_dataset.y[random_i]
            elif target_type == "random":
                y = full_dataset.y[rand_no_rep_seq[i]]
            else:
                sys.exit(" Data-to-process is not specified...")

            self.scenes.append((x, edge_index, edge_attr, y))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index) -> T_co:
        #TODO improve, send out the Data object directy, then in the rgcn_model.py file,
        # fix the "batch" object. basically the output of this function is the value of batch
        
        x, edge_list, edge_attr, y = self.scenes[index]
        return Data(x, edge_list, edge_attr, y), self.min_id + index


class SceneGraphDataset(Dataset):
    # TODO remove daaset_path
    def __init__(self, args, dataset_path):
        self.schema_path = args.schema_path
        self.dataset_name = args.source_type # IEPVQA, IEPVQA-DIS, IEPVQA-QA # REMOVE
        self.scenes_path = args.obj_ann_path
        self.data_to_process = args.target_type # features, random, randommore # REMOVE

        self.dataset_path = args.dataset_save_path

        self.x = []
        self.edge_attr = []
        self.edge_index = []
        self.y = []
        self.process()

    def __len__(self):
        return len(self.x)

    def get_reverse_attr_map(self, attribute_map):
        attr_values = [v for values in attribute_map.values() for v in values]
        return {v: i for i, v in enumerate(attr_values)}

    def transform_objects(self, objects):
        # CREATE NODES - One-hot encoding
        x_list = []
        for obj in objects:
            obj_feat = [0 for _ in range(len(self.attr2id))]
            for attr in self.attr_keys:
                # print('-------')
                # print(attr)
                # print(obj)
                # print(self.attr2id)
                # exit()
                obj_feat[self.attr2id[obj[attr]]] = 1
            x_list.append(obj_feat)

        return torch.tensor(x_list, dtype=torch.float)

    def transform_edges(self, relations):
         # CREATE EDGES -  One-hot encoding

        # TODO IDEA 1:
        # maybe add "ghost" edges to make the graph fully conected?
        # as such, the length of y would equal the number of edges (for CLEVR-GNN)

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
                    edge_feat = [0 for _ in range(len(self.rel2id))]
                    edge_feat[self.rel2id[rel_type]] = 1
                    edge_attr_list.append(edge_feat)
        edge_index = torch.tensor([edge_index_src, edge_index_tgt], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        return (edge_index, edge_attr)        

    def transform_outputs(self, index):
        raise NotImplementedError

    def process(self):

        # Check save file instead of recreating the dataset
        read_from_save = self.dataset_path != None and len(os.listdir(self.dataset_path)) != 0
        if read_from_save:
            # x = torch.load(f'{self.dataset_path}/x.pt')
            # edge_attr = torch.load(f'{self.dataset_path}/edge_attr.pt')
            # edge_index = torch.load(f'{self.dataset_path}/edge_index.pt')
            y = torch.load(f'{self.dataset_path}/y.pt')
            self.y = torch.unsqueeze(y, dim=1)

        # READ SCHEMA
        with open(self.schema_path) as f:
            schema = json.load(f)

        self.attr2id = self.get_reverse_attr_map(schema['attributes'])
        self.attr_keys = list(schema['attributes'].keys())

        self.rel2id = {v: i for i, v in enumerate(schema['relations'])}

        # READ THE SCENES DATA
        with open(self.scenes_path, 'r') as f:
            scenes = json.load(f)['scenes']

        # SELECT RELEVANT SUBSET
        if self.global_max_id is not None:
            scenes = scenes[:self.global_max_id]

        # ITERATE THROUGH THE SCENES
        for scene in tqdm(scenes):

            # >>>>>> INPUTS
            # nodes - One-hot encoding
            x = self.transform_objects(scene['objects'])
            self.x.append(x)
            # edges - One-hot encoding
            edge_index, edge_attr = self.transform_edges(scene['relationships'])
            self.edge_attr.append(edge_attr)
            self.edge_index.append(edge_index)

            # TODO - questions - encoding

            # >>>>>> OUTPUTS
            # NOTE only correct outputs, shuffling is in the datasubset
            if not read_from_save:
                y = self.transform_outputs(scene['image_index'])
                self.y.append(y)

            # #CREATE torch_geometric.Data.Data OBJECT TO SAVE
            # graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            # self.scenes.append(graph)

        # Save tensors to file, for quicker access in the future
        if self.dataset_path != None and len(os.listdir(self.dataset_path)) == 0:
            torch.save(torch.cat(self.y), f'{self.dataset_path}/y.pt')    


class SGDataset(SceneGraphDataset):
    def __init__(self, args):
        self.data_file_path = args.target_type
        dataset_path = f'{args.dataset_dir}/{args.target_type}'
        self.global_max_id = args.global_max_id

        
        self.target_path = args.target_path
        super().__init__(args, dataset_path)

    def process(self):
        super().process()

    def transform_outputs(self, index):
        # get features intermediate rep from save file
        if os.path.isdir(self.target_path):
            # Cannot make this an h5 file since number of features is inconsistent between images
            # one feature per object
            feature_path = f'{self.target_path}/{index}.pt'
            y = torch.load(feature_path)
            y.requires_grad = False
        else:
            with h5py.File(self.target_path, 'r') as f :
                y = torch.tensor(f['features'][index], dtype=torch.float)
                y = y.unsqueeze(0)
        # TODO handle relations through neuron values from multiple layers:
        # if self.data_to_process == "relations":
        
        #     layer_id = 2 # 1 for test set
        #     neuron_path = f'{neuron_dir_path}/{ind}.pt'
        #     neuron_data = torch.load(neuron_path)
        #     y = neuron_data['layers'][layer_id]
        return y


class SGQADataset(SceneGraphDataset):
    def __init__(self, args):
        # assert args.iep_answer_details != None
        # iep_specs = args.iep_answer_details
        # self.data_file_path = f'{args.target_type}-{iep_specs[1]}-{iep_specs[2]}-{iep_specs[3]}'
        # dataset_path = f'{args.dataset_dir}/{args.dataset_name}/{iep_specs[0]}'
        dataset_path = '' # TEMP
        self.global_max_id = args.global_max_id
        
        self.decoder = args.decoder

        # (1) Get all Questions
        with h5py.File(args.questions_path, 'r') as f_q:
            self.questions_q = self._dataset_to_tensor(f_q['questions'])
            self.image_ids_q = self._dataset_to_tensor(f_q['image_idxs'])
            question_ids_q = self._dataset_to_tensor(f_q['orig_idxs'])

        # This basically ensures that if we query f_q['image_idxs'] with the content of
        # f_a['question_id'], we will get a consistent result.
        for id_of_id, id in enumerate(question_ids_q):
            assert id_of_id == id
        
        # (2) Get all Answers
        with h5py.File(args.answers_path, 'r') as f_a:
            # 'scores' ignored for now
            self.question_ids_a = torch.tensor(f_a['question_ids'])
            self.results_a = torch.tensor(f_a['results']) # result id
        
        # (3) Get answer length
        with open(args.vocab_path, 'r') as f:
            vocab = json.load(f)
            self.answer_length = len(vocab['answer_token_to_idx'].keys())
            args.answer_length = self.answer_length

        # (4) Determine how long the encoding needs to be
        # NOTE Assuming that answer is shorter than question
        question_length = len(self.questions_q[0])
        assert self.answer_length <= question_length
        # encoding_length = max(answer_length, question_length)
        # pad_length = question_length-answer_length

        # (5) VALIDATION: Check how many questions there are per image
        num_questions_per_image = [0] * self.global_max_id
        for id in self.question_ids_a:
            im_id = self.image_ids_q[id.int()]
            if im_id >= self.global_max_id:
                continue
            num_questions_per_image[im_id] += 1
        
        num_qs_per_image = max(num_questions_per_image)
        for num_q in num_questions_per_image:
            # check that the number of questions per image is constant
            assert num_q == num_qs_per_image or num_q == 0

        super().__init__(args, dataset_path)


    def _dataset_to_tensor(self, dset, mask=None):
        arr = np.asarray(dset, dtype=np.int64)
        if mask is not None:
            arr = arr[mask]
        tensor = torch.tensor(arr, dtype=torch.long)
        return tensor
    
    def process(self):
        super().process()

    def transform_outputs(self, index):
        #Out is one-hot encoding of the questions and answers

        # Gather questions associated to index
        questions_for_ind_list = []
        for a_id, q_id_flt in enumerate(self.question_ids_a):
            q_id = q_id_flt.int()
            if self.image_ids_q[q_id] == index:
                # CREATE A COUNTER HERE FOR TESTING

                #Only results handled for now
                q = self.questions_q[q_id] # get QUESTION encoding (as int)
                a_raw = self.results_a[a_id]
                a = torch.zeros(q.size(), dtype=torch.int)
                a[int(a_raw.item())] = 1 # make ANSWER one-hot

                #create the [2, encoding_length] Tensor
                qa = torch.stack((q, a), dim=0)
                
                # append to a tensor that will eventually be of length num_qs_per_image
                questions_for_ind_list.append(qa)
        
        y = torch.stack(questions_for_ind_list).unsqueeze(0)
        # Should be [1, num_qs_per_image, 2, question_length]

        return y

