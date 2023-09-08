from datasets.mlpdataset import MLPDataLoader
from datasets.lstmdataset import LSTMDataLoader
from datasets.cnn1ddataset import CNN1DDataLoader
from datasets.cnn2ddataset import CNN2DDataLoader
from datasets.recordeddataset import SourceCountingDataLoader

from model.mlp import MLPModule
from model.lstm import LSTMModule
from model.cnn1d import CNN1DModule
from model.cnn2d import CNN2DModule

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    data_types = ["train", "validate", "test"]
    dataloaddict = {
        "mlp": MLPDataLoader,
        "lstm": LSTMDataLoader,
        "cnn1d": CNN1DDataLoader,
        "cnn2d": CNN2DDataLoader
    }
    moduledict = {
        "mlp": MLPModule,
        "lstm": LSTMModule,
        "cnn1d": CNN1DModule,
        "cnn2d": CNN2DModule
    }
    dataloadtype= dataloaddict[cfg.model]
    module= moduledict[cfg.model]
    
    if cfg.mode == "simulation":
        data_loaders = {}

        for data_type in data_types:
            data_loader = dataloadtype(getattr(cfg, data_type), window_size= cfg.dataloader.window_size, stride= cfg. dataloader.stride, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers)
            data_loaders[data_type] = data_loader

        checkpoint_callback= ModelCheckpoint(
            dirpath="Checkpoint",
            filename="weights={epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}",
            monitor=cfg.trainer.checkpoint.monitor,
            save_top_k=1, 
            mode=cfg.trainer.checkpoint.mode,
            verbose=True
            )

        model = module(cfg.module.loss, cfg.module.weight_decay, cfg.module.learning_rate, cfg.module.scheduler)

        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=cfg.trainer.epochs,
            logger=True,
            callbacks= [checkpoint_callback, EarlyStopping(monitor= cfg.trainer.early_stopping.monitor, patience= cfg.trainer.early_stopping.patience)], 
            num_sanity_val_steps=0,
            accumulate_grad_batches=1,
            gradient_clip_val=0.5
        )
        
    
        trainer.fit(model, data_loaders["train"], data_loaders["validate"])
        trainer.test(dataloaders=data_loaders["test"], ckpt_path=checkpoint_callback.best_model_path)
    
    elif cfg.mode == "recorded":
        mlp_model=MLPModule.load_from_checkpoint(cfg.checkpoint_path, loss=cfg.module.loss, weight_decay=cfg.module.weight_decay, learning_rate=cfg.module.learning_rate, scheduler=cfg.module.scheduler)
        test_data_loader = SourceCountingDataLoader(cfg.test, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers)
        trainer = pl.Trainer()
        trainer.test(model=mlp_model, dataloaders=test_data_loader)
        predictions= mlp_model.predictions
        print(predictions)
        print("Model tested")

    else:
        print("Invalid mode")
        
if __name__ == "__main__":
    main()
