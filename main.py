from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
from model.cnn import CNN1DModule
from model.rnn import RNNModule
import pytorch_lightning as pl
import torch
from torchviz import make_dot
import os
if __name__ == "__main__":

    # logs_directory = '/SCNN/lightning_logs/'
    
    # # Delete all files in the logs directory
    # for filename in os.listdir(logs_directory):
    #     file_path = os.path.join(logs_directory, filename)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.unlink(file_path)
    #     except Exception as e:
    #         print(f"Error deleting {file_path}: {e}")

    train = '/dataset/train/'  
    validate = '/dataset/validate/'  
    test = '/dataset/test/'  

    batch_size = 25
    epochs = 5
    num_workers = 4
    training_data = SourceCountingDataLoader(train, batch_size=batch_size, num_workers=num_workers)
    validating_data = SourceCountingDataLoader(validate, batch_size=batch_size, num_workers=num_workers)
    testing_data = SourceCountingDataLoader(test, batch_size=batch_size, num_workers=num_workers)

    # Train the model
    print("Data loaded")
    mlp_model = MLPModule()
    print("Model loaded")
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=epochs,
        logger=True
    )
    trainer.fit(mlp_model, training_data, validating_data)
    print("Model trained")

    # Visualize the MLP model architecture
    dummy_input = torch.rand(batch_size, 200)  # Adjust input size accordingly
    dot = make_dot(mlp_model(dummy_input), params=dict(mlp_model.named_parameters()))
    dot.render("mlp_model", format="png")  # Save the visualization as a PNG file
    print("Model architecture visualized")

    # Test the trained model
    trainer.test(dataloaders=testing_data)


    
