import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import json
import torch.nn as nn
# GET THE GCC_PHAT OF SIGNALS AND THE NUMBER OF SOURCES
# LOAD THE GCC_PHAT OF SIGNALS


#CREATE DATASET

class SourceCountingDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.processed= torch.load(self.processed_dir)
    def __len__(self):
        return len(self.processed["gcc_phat"])

    def __getitem__(self, index):
        gcc_phat_tensor = self.processed["gcc_phat"][index]
        target=self.processed["targets"][index]
        target_class = nn.functional.one_hot(target.clone().detach(), num_classes=2).float()
        return gcc_phat_tensor, target_class    
        #returning gcc_phat and number of sources for each sample


# LOAD DATASET
class SourceCountingDataLoader(DataLoader):
    def __init__(self, processed_dir, batch_size, shuffle=False, num_workers=5):
        dataset = SourceCountingDataset(processed_dir)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=num_workers)
   

from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
from model.cnn import CNN1DModule
from model.rnn import RNNModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import hydra
from omegaconf import DictConfig


def get_data(processed_dir, types, batch_size, num_workers):
    data=[]
    print(types, batch_size, num_workers)
    for data_type in types:
        print(data_type)
        data_path = os.path.join(processed_dir, f"{data_type}.pt")
        data.append(SourceCountingDataLoader(data_path, batch_size=batch_size, num_workers=num_workers))
    return data

@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
   
    processed_dir=cfg.output_folder  
    data_types=["train","validate","test"]
    train, validate, test = get_data(processed_dir, data_types, cfg.batch_size, cfg.num_workers)
    mlp_model = MLPModule()

    # icheckpoint_callback = ModelCheckpoint(
    #     dirpath="checkpoints",
    #     filename="best-checkpoint-i",
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val_loss",
    #     mode="min"
    # )

    itrainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=cfg.epochs,
        logger=True
        # callbacks=[
        #     icheckpoint_callback,
        #     EarlyStopping(monitor="val_loss", patience=3)]    
            )
    
    itrainer.fit(mlp_model, train, validate)
    # best_model_path_i = itrainer.checkpoint_callback.best_model_path
    # best_model_i = MLPModule.load_from_checkpoint(best_model_path_i)
    # best_model_i.eval()
    itrainer.test(dataloaders=test, ckpt_path="best")

    # # Fine-tune for [0,3] sources
    # type=["trainf","validatef","testf"]
    # train, validate, test = get_data(processed_dir, type, cfg.batch_size, cfg.num_workers)
    # mlp_finetune= MLPModule()
    # mlp_finetune.load_state_dict(best_model_i.state_dict())
    # # Load the best model from the second stage and continue training
    # fcheckpoint_callback = ModelCheckpoint(
    #     dirpath="checkpoints",
    #     filename="best-checkpoint-f",
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val_loss",
    #     mode="min"
    # )
    # ftrainer = pl.Trainer(
    #     log_every_n_steps=1,
    #     max_epochs=cfg.epochs,
    #     logger=True,
    #     callbacks=[
    #         fcheckpoint_callback,
    #         EarlyStopping(monitor="val_loss", patience=3)]
    #         )
    # ftrainer.fit(mlp_finetune, train, validate)
    # best_model_path_f = ftrainer.checkpoint_callback.best_model_path
    # best_model_f = MLPModule.load_from_checkpoint(best_model_path_f)
    # best_model_f.eval()
    # ftrainer.test(dataloaders=test, ckpt_path="best")


if __name__ == '__main__':
    main()
