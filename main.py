from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    data_types = ["train", "validate", "test"]
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    num_workers = cfg.num_workers
    
    data_loaders = {}

    for data_type in data_types:
        data_loader = SourceCountingDataLoader(getattr(cfg, data_type), batch_size=batch_size, num_workers=num_workers)
        data_loaders[data_type] = data_loader

    # Train the model
    print("Data loaded")
    mlp_model = MLPModule()
    print("Model loaded")
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=epochs,
        logger=True
    )
    
    for data_type in data_types:
        trainer.fit(mlp_model, data_loaders["train"], data_loaders["validate"])
        print(f"Model trained for {data_type}")

    # Test the trained model
    trainer.test(dataloaders=data_loaders["test"], ckpt_path='best')


if __name__ == "__main__":
    main()
