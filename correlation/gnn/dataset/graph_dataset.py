from distutils.errors import LinkError
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
from pathlib import Path

# TODO get rid of these
neuron_dir_path = "_results/4000images/neurons"
feature_dir_path = "_results/4000images/node2feature"
# TODO make this the dataset_h5_path
# iepvqa_h5_path = "_data/iepvqa/val_features_0_3000.h5" 
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
        self.dataset_h5_path = args.dataset_h5_path
        self.iep_a_spec = args.iep_answer_details

        assert self.dataset_name != "IEPVQA-QA" or (self.dataset_name == "IEPVQA-QA" and self.iep_a_spec != None )

        if self.dataset_name == "IEPVQA-QA":
            subfolder_name = f'{self.dataset_name}/{self.iep_a_spec[0]}'
            self.data_file_path = f'{self.data_to_process}-{self.iep_a_spec[1]}-{self.iep_a_spec[2]}-{self.iep_a_spec[3]}'
        else:
            subfolder_name = self.dataset_name
            self.data_file_path = self.data_to_process

        super().__init__(f'{args.dataset_dir}/{subfolder_name}')
        self.data, self.slices, = torch.load(self.processed_paths[0]) # this is specific to "InMemoryDatasets"        

    def get_reverse_attr_map(self, attribute_map):
        attr_values = [v for values in attribute_map.values() for v in values]
        return {v: i for i, v in enumerate(attr_values)}

    @property
    def processed_file_names(self):
        return [f'data-{self.data_file_path}.pt']

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

        # Currently hard-coded limit of 3000 scenes for IEPVQA
        # because we don't need all 15000 validation images
        # REMEMBER that this is for the full dataset.
        # We can select subset through the cmd-line args when creating
        # the SceneGraphDataSeubset object
        if self.dataset_name.startswith('IEPVQA'):
            if self.iep_a_spec != None:
                scenes = scenes[self.iep_a_spec[1]:self.iep_a_spec[2]]
            else:
                if self.dataset_name == 'IEPVQA':
                    scenes = scenes[:3000]
                elif self.dataset_name == 'IEPVQA-DIS':
                    scenes = scenes[:1000]

        if self.dataset_name == 'IEPVQA-QA':
            # do some measurements
            import sys
            sys.path.append('../clevr-iep')
            from iep.data import _dataset_to_tensor

            # --------------------------
            # THESE ARE ALL THE QUESTION
            path_to_vqa_questions="../clevr-iep/data/questions/val_questions_0_3000.h5"
            f_q = h5py.File(path_to_vqa_questions, 'r')
            
            questions_q = _dataset_to_tensor(f_q['questions'])
            image_ids_q = _dataset_to_tensor(f_q['image_idxs'])
            question_ids_q = _dataset_to_tensor(f_q['orig_idxs'])

            # This basically ensures that if we query f_q['image_idxs'] with the content of
            # f_a['question_id'], we will get a consistent result.
            for id_of_id, id in enumerate(question_ids_q):
                assert id_of_id == id

            # ----------------------------------------
            # THESE ARE ONLY SELECTED ANSWERS (SUBSET)
            # TODO big assumptions here. regarding path_to_answers
            # number of images, and question interval should be given as input
            path_to_vqa_answers=f"../clevr-iep/data/answers/resnet101/{self.iep_a_spec[0]}_{self.iep_a_spec[1]}_{self.iep_a_spec[2]}__{self.iep_a_spec[3]}.h5"
            # path_to_vqa_answers=f"../clevr-iep/data/answers/resnet101/{self.iep_a_spec}_0_3000__10.h5"
            f_a = h5py.File(path_to_vqa_answers, 'r')

            scores_a = torch.Tensor(f_a['scores'])
            question_ids_a = torch.Tensor(f_a['question_ids'])
            results_a = torch.Tensor(f_a['results'])

            # ---------------------
            # 3. are sizes fixed? same format? can we concatenate them in y?

            question_length = len(questions_q[0])
            answer_length = len(scores_a[0][0])

            # ASSUME TODO that answer is shorter than question
            assert answer_length <= question_length
            # encoding_length = max(answer_length, question_length)
            # pad_length = question_length-answer_length

            # --------------------------
            # 0. see how many questins there are per image? maybe pad with nones?
            # ASSUME TODO number of questions per image is constant
            num_questions_per_image = [0] * 3000 # 3000 is the max number of images handled for now
            for id in question_ids_a:
                num_questions_per_image[image_ids_q[id.int()]] += 1
            
            num_qs_per_image = max(num_questions_per_image)
            for num_q in num_questions_per_image:
                assert num_q == num_qs_per_image or num_q == 0

            # -----------------------------
            # create random sequence, iterate over questions
            random_sequence = list(range(len(question_ids_a)))
            
            f_q.close()
            f_a.close()
        else:
            # -----------------------------
            # create random sequence, iterate over scenes
            random_sequence = list(range(len(scenes)))

        # Create random ordering. Currently only working for IEP:
        # TODO introduce a deterministic seed, for reproductibility
        random.shuffle(random_sequence)


        # ITERATE THROUGH THE SCENES
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
                # get ground truth feature values from feature extractor of scene objects
                if self.data_to_process == "relations":
                    neuron_path = f'{neuron_dir_path}/{ind}.pt'
                    neuron_data = torch.load(neuron_path)
                    y = neuron_data['layers'][layer_id]
                elif self.data_to_process == "features":
                    feature_path = f'{feature_dir_path}/{ind}.pt'
                    y = torch.load(feature_path)
                elif self.data_to_process.startswith("random"):
                    feature_path = f'{feature_dir_path}/{ind}.pt'
                    features = torch.load(feature_path)
                    lb = torch.min(features)
                    ub = torch.max(features)
                    y = (ub-lb) * torch.rand(np.shape(features)) + lb
                else:
                    print(" Data-to-process is not specified...")
                    exit(1)
            elif self.dataset_name == 'IEPVQA' or self.dataset_name == "IEPVQA-DIS":
                # get ground truth features value from the VQA intermediate representation
                # NOTE: self.dataset_h5_path is derived from clevriep/scripts/0-run-feature-extraction
                with h5py.File(self.dataset_h5_path, 'r') as f :
                    if self.data_to_process == "features":
                        y = torch.FloatTensor(f['features'][ind])
                        y = y.unsqueeze(0)
                    elif self.data_to_process.startswith("randomwithreplacement"):
                        random_ind = random.randint(0, len(scenes)-1)
                        y = torch.FloatTensor(f['features'][random_ind])
                        y = y.unsqueeze(0) 
                    elif self.data_to_process.startswith("random"):
                        real_feat = torch.FloatTensor(f['features'][ind])
                        y = torch.FloatTensor(f['features'][random_sequence[ind]])
                        assert y.size() == real_feat.size()
                        y = y.unsqueeze(0)
                    else:
                        print(" Data-to-process is not specified...")
                        exit(1)
                f.close()
            elif self.dataset_name == 'IEPVQA-QA':

                # --------------------------
                # THESE ARE ALL THE QUESTION
                # TODO make this "dataset_h5_path" in yaml
                f_q = h5py.File(path_to_vqa_questions, 'r')
                
                questions_q = _dataset_to_tensor(f_q['questions'])
                image_ids_q = _dataset_to_tensor(f_q['image_idxs'])
                question_ids_q = _dataset_to_tensor(f_q['orig_idxs'])

                # ----------------------------------------
                # THESE ARE ONLY SELECTED ANSWERS (SUBSET)
                # TODO big assumptions here. regarding path_to_answers
                # number of images, and question interval should be given as input
                f_a = h5py.File(path_to_vqa_answers, 'r')

                scores_a = torch.Tensor(f_a['scores'])
                question_ids_a = torch.Tensor(f_a['question_ids'])
                results_a = torch.Tensor(f_a['results'])

                # ---------------------
                # Go through every answer,and only consider the ones related to the the current image

                questions_for_ind_list = []
                for a_id, q_id_flt in enumerate(question_ids_a):
                    q_id = q_id_flt.int()
                    if image_ids_q[q_id] == ind:
                        # CREATE A COUNTER HERE FOR TESTING

                        # SCORES
                        if self.data_to_process.startswith("scores-"):
                            q = questions_q[q_id].to(torch.float) # get QUESTION encoding (transform to floats)
                            data_spec = self.data_to_process[len("scores-"):]
                            if data_spec == "features":
                                a_raw = scores_a[a_id][0] # get ANSWER scores (as float)
                            elif data_spec.startswith("randomwithreplacement"):
                                random_ind = random.randint(0, len(question_ids_a)-1)
                                a_raw = scores_a[random_ind][0] # get ANSWER scores (as float)
                            elif data_spec.startswith("random"):
                                a_raw = scores_a[random_sequence[a_id]][0] # get ANSWER scores (as float)
                            else:
                                print(" Data-to-process suffix is invalid (should be \"features\" or \"random\" or \"randomWithReplacement\")")
                                exit(1)

                            a = torch.full(q.size(), float('-inf'))
                            a[:len(a_raw)] = a_raw # pad ANSWER scores with minus infinities

                        # RESULTS
                        elif self.data_to_process.startswith("results-"):
                            q = questions_q[q_id] # get QUESTION encoding (as int)
                            data_spec = self.data_to_process[len("results-"):]
                            if data_spec == "features":
                                a_raw = results_a[a_id] # get ANSWER index
                            elif data_spec.startswith("randomwithreplacement"):
                                random_ind = random.randint(0, len(question_ids_a)-1)
                                a_raw = results_a[random_ind] # get ANSWER index
                            elif data_spec.startswith("random"):
                                a_raw = results_a[random_sequence[a_id]] # get ANSWER index
                            else:
                                print(" Data-to-process suffix is invalid (should be \"features\" or \"random\" or \"randomWithReplacement\")")
                                exit(1)

                            a = torch.zeros(q.size())
                            a[a_raw] = 1 # make ANSWER one-hot
                        else:
                            print(" Data-to-process prefix is invalid (should be \"scores-\" or \"results-\")")
                            exit(1)

                        #create the [2, encoding_length] Tensor
                        qa = torch.stack((q, a), dim=0)
                        # print(qa.dtype)
                        # print(qa)
                        # print(qa.size())
                        
                        # append to a tensor that wiull eventually be of length num_qs_per_image
                        questions_for_ind_list.append(qa)
                
                y = torch.stack(questions_for_ind_list).unsqueeze(0)
                # Should be [1, num_qs_per_image, 2, question_length]

                # print(qs_for_ind)

                # print(y.size())
                # print(y.dtype)
                # # exit()
                
                f_q.close()
                f_a.close()
            else:
                print(f"Dataset root directory not set. {self.dataset_name} is invalid.")
                exit(1)

            #CREATE torch_geometric.Data.Data OBJECT TO SAVE
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(graph)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
