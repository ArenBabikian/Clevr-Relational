import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from correlation.gnn.dataset.graph_dataset import SceneGraphDatasubset, SGDataset, SGQADataset
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

def createDataLoader(self, dataset):
    return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=graph_collate,
            pin_memory=True
        )

DATASET_MAP = {
    'CLEVR-GNN': SGDataset,
    'IEPVQA': SGDataset,
    'IEPVQA-STEM': SGDataset,
    'IEPVQA-Q': SGQADataset,
    'IEPVQA-Q-STEM': SGQADataset,

    'IEPVQA-DIS': SGDataset
}
dataset_2_encoder = {
    'CLEVR-GNN':{'gat', 'rgcn', 'rgcn2'},
    'IEPVQA':{'gatiep'},
    'IEPVQA-STEM':{'gatstem'},
    'IEPVQA-Q':{'gatiep'},
    'IEPVQA-Q-STEM':{'gatstem'}
}

class SceneGraphDatasetModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.id_min = args.min_id if hasattr(args, "min_id") else 0
        self.id_cutoff = self.id_min + args.train_size
        self.id_max = args.max_id if hasattr(args, "max_id") else None
        assert self.id_min < self.id_cutoff
        assert self.id_max == None or self.id_cutoff <= self.id_max

        # INPUT VALIDATION
        if args.source_type not in dataset_2_encoder.keys():
            exit(f'{args.source_type} not supported')
        if args.encoder not in dataset_2_encoder[args.source_type]:
            exit(f'\'{args.encoder}\' not supported for {args.source_type}')

        self.full_dataset = DATASET_MAP[args.source_type](args)
        self.target_type = args.target_type

    def train_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, self.target_type, min_id=self.id_min, max_id=self.id_cutoff)
        return createDataLoader(self, dataset)

    # TODO fix this
    # (1) this may point to the 1000 original test images used for CLEVR
    # (2) we might want to try to implement a run_test.py file to independently run the test.
    def test_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, self.target_type, min_id=self.id_cutoff, max_id=self.id_max)
        return createDataLoader(self, dataset)

    def val_dataloader(self):
        dataset = SceneGraphDatasubset(self.full_dataset, self.target_type, min_id=self.id_cutoff, max_id=self.id_max)
        return createDataLoader(self, dataset)
