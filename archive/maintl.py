from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
from archive.multiclass.multiclassifier import MultiClassifictionModule
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    data_types = ["train", "validate", "test"]
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    num_workers = cfg.num_workers
    repeat= cfg.repeat
    data_loaders = {}
    data_loaders_2 = {}

    a={
        "train": "/finaldatasetalt1/train/",
        "validate" : "/finaldatasetalt1/validate/", 
        "test" : "/finaldatasetalt1/test/"    
    }
    b={
        "train": "/finaldatasetalt2/trains/",
        "validate" : "/finaldatasetalt2/validate/", 
        "test" : "/finaldatasetalt2/test/"    
    }
    for data_type in data_types:
        data_loader = SourceCountingDataLoader(a[data_type], batch_size=batch_size, num_workers=num_workers)
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

    mlp_model = MLPModule()

    trainer = pl.Trainer(
        log_every_n_steps=10,
        max_epochs=5,
        logger=True,
        callbacks= [checkpoint_callback, EarlyStopping(monitor="val_loss", patience=3)], 
        num_sanity_val_steps=0,
        gradient_clip_val=0.3,
        accumulate_grad_batches=1
    )
    
  
    trainer.fit(mlp_model, data_loaders["train"], data_loaders["validate"])
    print("Model trained")

    # Test the trained model
    trainer.test(dataloaders=data_loaders["test"], ckpt_path=checkpoint_callback.best_model_path)
    for data_type in data_types:
            data_loader = SourceCountingDataLoader(b[data_type], batch_size=batch_size, num_workers=num_workers)
            data_loaders_2[data_type] = data_loader
            
    # Fine tuning
    print("Fine tuning")
    mlptl = MLPModule.load_from_checkpoint(checkpoint_callback.best_model_path)

    checkpoint_callback_2= ModelCheckpoint(
            dirpath="/Checkpoint",
            filename="weights={epoch:02d}-{val_loss:.2f}",
            monitor="val_loss", 
            save_top_k=1, 
            mode="min",
            verbose=True
            )
  
    trainer2 = pl.Trainer(
        log_every_n_steps=10,
        max_epochs=epochs,
        logger=True,
        callbacks= [checkpoint_callback_2, EarlyStopping(monitor="val_loss", patience=3)], 
        num_sanity_val_steps=0,
        gradient_clip_val=0.3,
        accumulate_grad_batches=1
    )
    
    trainer2.fit(mlptl, data_loaders_2["train"], data_loaders_2["validate"])
    print(f"Model trained")

    # Test the trained model
    trainer.test(dataloaders=data_loaders_2["test"], ckpt_path=checkpoint_callback_2.best_model_path)

if __name__ == "__main__":
    main()