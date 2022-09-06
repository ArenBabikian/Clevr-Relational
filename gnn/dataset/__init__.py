from torch.utils.data import DataLoader

from .clevr_dataset import ClevrSceneDataset
import pytorch_lightning as pl


class SceneDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        dataset = ClevrSceneDataset(self.args.obj_ann_path, self.args.schema_path, max_scene_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = ClevrSceneDataset(self.args.obj_ann_path, self.args.schema_path, min_scene_id=self.args.train_size)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader
