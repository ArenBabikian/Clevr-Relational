import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from correlation.gnn.dataset.graph_dataset import SceneGraphDataset, SceneGraphDatasubset
import pytorch_lightning as pl

def graph_collate(batch):
    """
        this function replaces the default collate function of pytorch to handle the PyG Data object
        batch: list of [Data, int]
        return PyG Batch, tensor
    """
    data_list = [sample[0] for sample in batch]
    indexes = torch.LongTensor([sample[1] for sample in batch])
    return Batch.from_data_list(data_list), indexes
    
class SceneGraphDatasetModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.full_dataset = SceneGraphDataset('_datasets/SCENES', self.args.obj_ann_path, self.args.schema_path, self.args.intermediate_gt)

    def train_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, max_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader

    # TODO fix this
    # (1) this may point to the 1000 original test images used for CLEVR
    # (2) we might want to try to implement a run_test.py file to independently run the test.
    def test_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, min_id=self.args.train_size)

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
        dataset = SceneGraphDatasubset(self.full_dataset, min_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader
