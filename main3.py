from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
from model.singleclassifier import ClassifierModule
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    data_types = ["train", "validate", "test"]
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    num_workers = cfg.num_workers
    repeat= cfg.repeat
    data_loaders = {}

    for data_type in data_types:
        data_loader = SourceCountingDataLoader(getattr(cfg, data_type), batch_size=batch_size, num_workers=num_workers)
        data_loaders[data_type] = data_loader
    test_prev_prediction= None
    # Train the model
    print("Data loaded")
    results= [-1]*1000
    for i in range(3):
        checkpoint_callback= ModelCheckpoint(
            dirpath=f"/Checkpoint/{i}",
            filename="weights={epoch:02d}-{val_loss:.2f}",
            monitor="val_loss", 
            save_top_k=1, 
            mode="min",
            verbose=True
            )
        mlp_model = ClassifierModule(i)

        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=epochs+i,
            logger=True,
            callbacks= [checkpoint_callback], 
            num_sanity_val_steps=0
        )

        trainer.fit(mlp_model, data_loaders["train"], data_loaders["validate"])
        print(f"Model trained for {i} or less sources")
       
        print("Predictions recorded")
        trainer.test(dataloaders=data_loaders["test"], ckpt_path=checkpoint_callback.best_model_path)
        test_prev_prediction = mlp_model.test_predictions
        print(test_prev_prediction)
        pred=torch.cat(test_prev_prediction, dim=0).squeeze().tolist()
        for j, test_pred in enumerate(pred):
            if test_pred<0.5 and results[j]==-1:
                results[j]=i
            if i==3 and results[j]==-1:
                results[j]=3
        
    targets=[]
    for a in range(1000):
        targets.append(data_loaders["test"].dataset[a][1].item())
    
    matching_count = sum(1 for val1, val2 in zip(targets, results) if val1 == val2)
    # Calculate the percentage of matching elements
    percentage_matching = (matching_count / len(targets)) * 100
    print("="*20)
    print(percentage_matching)
    print("="*20)
    
if __name__ == "__main__":
    main()