from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .clevr_dataset import ClevrSceneDataset
import pytorch_lightning as pl
import torch


def graph_collate(batch):
    """
        this function replaces the default collate function of pytorch to handle the PyG Data object
        batch: list of [Data, int]
        return PyG Batch, tensor
    """
    data_list = [sample[0] for sample in batch]
    indexes = torch.LongTensor([sample[1] for sample in batch])
    return Batch.from_data_list(data_list), indexes


class SceneDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        dataset = ClevrSceneDataset(self.args.obj_ann_path, self.args.schema_path, self.args.edge_transformer,
                                    max_scene_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        dataset = ClevrSceneDataset(self.args.obj_ann_path, self.args.schema_path, self.args.edge_transformer,
                                    min_scene_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = ClevrSceneDataset(self.args.obj_ann_path, self.args.schema_path, self.args.edge_transformer,
                                    min_scene_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader
