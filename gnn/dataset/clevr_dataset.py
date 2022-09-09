from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import json
import torch
from torch_geometric.data import Data


class ClevrSceneDataset(Dataset):
    def __init__(self, obj_ann_path, schema_path, edge_transformer, min_scene_id=None, max_scene_id=None):
        with open(obj_ann_path) as f:
            scenes = json.load(f)['scenes']

        with open(schema_path) as f:
            schema = json.load(f)

        self.attr2id = self.get_reverse_attr_map(schema['attributes'])
        self.num_attr = len(self.attr2id)
        self.attr_keys = list(schema['attributes'].keys())

        self.rel2id = {v: i for i, v in enumerate(schema['relations'])}
        self.num_rels = len(self.rel2id)
        self.edge_transformer = edge_transformer
        self.scenes = []
        self.scene_ids = []

        if min_scene_id is None:
            min_scene_id = 0
        if max_scene_id is None:
            max_scene_id = len(scenes)

        for i, scene in enumerate(scenes[min_scene_id: max_scene_id]):
            self.scenes.append(self.transform_scene(scene))
            self.scene_ids.append(min_scene_id + i)

    def get_reverse_attr_map(self, attribute_map):
        attr_values = [v for values in attribute_map.values() for v in values]
        return {v: i for i, v in enumerate(attr_values)}

    def transform_scene(self, scene):
        # transform the scene into node features(x), edge list and edge features
        x = self.transform_objects(scene['objects'])
        # here we use multi-hot encoding of edge features for both GAT or RGCN case to make the reconstruction easier
        edge_list, edge_features = self.transform_edge_features(scene['relationships'])

        return x, edge_list, edge_features

    def transform_objects(self, objects):
        x = []
        for obj in objects:
            obj_feat = [0 for _ in range(self.num_attr)]
            for attr in self.attr_keys:
                obj_feat[self.attr2id[obj[attr]]] = 1
            x.append(obj_feat)
        return x

    def transform_edge_features(self, relationships):
        edge_map = {}
        for rel, edges in relationships.items():
            for target, sources in enumerate(edges):
                for source in sources:
                    edge_feat = edge_map.get((source, target), [0 for _ in range(self.num_rels)])
                    edge_feat[self.rel2id[rel]] = 1
                    edge_map[(source, target)] = edge_feat

        edge_list = []
        edge_features = []
        for edge, feature in edge_map.items():
            edge_list.append(edge)
            edge_features.append(feature)

        return edge_list, edge_features

    def transform_edge_types(self, relationships):
        edge_list = []
        edge_types = []
        for rel, edges in relationships.items():
            rel = self.rel2id[rel]
            for target, sources in enumerate(edges):
                for source in sources:
                    edge_list.append((source, target))
                    edge_types.append(rel)
        return edge_list, edge_types

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index) -> T_co:
        x, edge_list, edge_features = self.scenes[index]
        x, edge_list, edge_features = torch.FloatTensor(x), torch.LongTensor(edge_list).T, \
                                      torch.FloatTensor(edge_features)

        return Data(x, edge_list, edge_features, edge_features), self.scene_ids[
            index]
