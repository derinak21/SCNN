from datasets.dataset import SourceCountingDataLoader
from model.mlp import MLPModule
import pytorch_lightning as pl
if __name__ == "__main__":
    train = '/datasets/train/'  
    validate = '/datasets/validate/'  
    test = '/datasets/test/'  

    output_directory = "path/to/output/directory"
    batch_size = 10
    epochs= 5
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
        max_epochs=5,
        logger=True 
    )    
    trainer.fit(mlp_model, training_data, validating_data)
    print("Model trained")
    # Test the trained model
    trainer.test(dataloaders=testing_data)


    
