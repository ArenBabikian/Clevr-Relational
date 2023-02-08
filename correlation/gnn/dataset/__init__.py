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
    data_list = []
    num_obj_seq = []
    indexes = []
    for sample in batch:
        data_list.append(sample[0])
        num_obj_seq.append(len(sample[0].x))
        indexes.append(sample[1])
    return Batch.from_data_list(data_list), torch.IntTensor(num_obj_seq), torch.LongTensor(indexes)
    
class SceneGraphDatasetModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.id_min = args.min_id if hasattr(args, "min_id") else 0
        self.id_cutoff = self.id_min + args.train_size
        self.id_max = args.max_id if hasattr(args, "max_id") else None
        assert self.id_min < self.id_cutoff
        assert self.id_max == None or self.id_cutoff <= self.id_max

        self.full_dataset = SceneGraphDataset(args)

    def train_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, min_id=self.id_min, max_id=self.id_cutoff)

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
        dataset = SceneGraphDatasubset(self.full_dataset, min_id=self.id_cutoff, max_id=self.id_max)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True,
            drop_last=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, min_id=self.id_cutoff, max_id=self.id_max)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True
        )
        return dataloader
