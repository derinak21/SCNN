# SCNN

This repository includes the SYDRA - Synthetic Datasets for Room Acoustics repository in order to create datasets by simulating a room. 

The dataset can be created by `python sydra/main.py dataset_dir=/path/to/dataset n_samples=1000`. 

A few changes has been made to SYDRA including giving a range for the number of sources to be selected in and plotting of the simulated room.

The cross correlation can be plotted by running `python sydra/sydra/visualization/cross_correlation.py /path/to/dataset/metadata.json /path/to/dataset/samples` and room can be plotted by running `python sydra/sydra/visualization/room_plotter.py /path/to/dataset/metadata.json`.

The `dataset.py` file creates a dataset using the cross correlation function as the input and the number of sources from the metadata as the output. 

The neural network model is defined in the `mlp.py`.

The loss functions are defined in `loss.py` and can be selected.

Change the directories to the datasets in the `main.py`

In order to create, train, validate and test a MLP neural network for voice activation, run `python main.py`.

