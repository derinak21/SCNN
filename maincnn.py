from datasets.dataset2d import SourceCountingDataLoader
from model.mlp import MLPModule
from model.rnn import RNNModule
from model.cnn import CNNModule
from model.lstm import LSTMModule
from model.multiclassifier import MultiClassifictionModule
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    data_types = ["train", "validate", "test"]
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    num_workers = cfg.num_workers
    window_size= cfg.window_size
    stride= cfg.stride
    data_loaders = {}

    for data_type in data_types:
        data_loader = SourceCountingDataLoader(data_type, getattr(cfg, data_type), window_size=window_size, stride= stride, batch_size=batch_size, num_workers=num_workers)
        data_loaders[data_type] = data_loader

    #save the best model with lowest validation loss
    checkpoint_callback= ModelCheckpoint(
        dirpath="/Checkpoint",
        filename="weights={epoch:02d}-{val_loss:.2f}",
        monitor="val_loss", 
        save_top_k=1, 
        mode="min",
        verbose=True
        )
    # Train the model
    print("Data loaded")

    cnn_model = CNNModule()

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=epochs,
        logger=True,
        callbacks= [checkpoint_callback, EarlyStopping(monitor="val_loss", patience=3)], 
        num_sanity_val_steps=0
    )
    
  
    trainer.fit(cnn_model, data_loaders["train"], data_loaders["validate"])
    print(f"Model trained")

    # Test the trained model
    trainer.test(dataloaders=data_loaders["test"], ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()