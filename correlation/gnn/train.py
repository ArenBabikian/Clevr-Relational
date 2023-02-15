import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers, Trainer
import pytorch_lightning as pl
import torch_geometric
import argparse
from correlation.gnn.config import SceneReconstructionConfig
from correlation.gnn.dataset import SceneGraphDatasetModule

from correlation.gnn.models.gnn_model import SceneConstructionModule

def main(args):
    if args.seed is not None:
        pl.seed_everything(seed=args.seed)
        torch_geometric.seed_everything(seed=args.seed)

    train_data = SceneGraphDatasetModule(args)
    model = SceneConstructionModule(args)
    checkpoint_callback = ModelCheckpoint(monitor="loss/val", dirpath=args.run_dir)
    # logger = pl_loggers.TensorBoardLogger(args.run_dir + "/logs/", name=f'{args.encoder}/{args.target_type}')
    logger = pl_loggers.TensorBoardLogger(f'{args.run_dir}/{args.source_type}', name=f'{args.encoder}/{args.target_type}')

    
    trainer = Trainer(
        fast_dev_run=args.dev,
        logger=logger,
        gpus=args.gpu,
        weights_summary=None,
        deterministic=args.seed is not None,
        log_every_n_steps=1,
        check_val_every_n_epoch=args.val_every_n_epoch,
        max_epochs=args.max_epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=args.precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(model, train_data)
    trainer.test(model=model, dataloaders = train_data.test_dataloader())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as f:
        dataMap = yaml.safe_load(f)

    config = SceneReconstructionConfig(dataMap)
    main(config)

