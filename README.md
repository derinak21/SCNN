# SCNN (Speaker Counting Neural Network)

This repository counts number of speakers in a room with a range of [0,2] using a neural network.  

## Setup
This repository can be cloned using the following command: 

1. `git clone https://github.com/derinak21/SCNN.git --recurse-submodules`

The dependencies can be installed using the following commands:

2. For Windows: 

 `.\install_dependencies.bat`

For MacOS and Linux:

 `./install_dependencies.sh`

## Creating Simulation Datasets
This repository includes the `SYDRA` - Synthetic Datasets for Room Acoustics repository in order to create datasets by simulating a room. 

The dataset can be created inside the repository by the following command:

3. `python sydra/main.py dataset_dir= dataset n_samples=1000`. 

The generalized cross correlation can be plotted by running `python sydra/sydra/visualization/cross_correlation.py dataset/metadata.json /path/to/dataset/samples` and room can be plotted by running `python sydra/sydra/visualization/room_plotter.py dataset/metadata.json`.


## Neural Networks 

From the `configuration` file, the mode can be selected.
Simulation mode uses the simulated data to train, validate and test the neural network.

Recorded mode uses a single `recording.wav` file inside `recording` folder and creates a dataset by splitting the audio file into 3 second audio files. 

For the recording mode, change the checkpoint_path in `configuration` file for the wanted weightings. 

A sample dataset was included in the repo under `data` folder. It contains datasets for training, validation and testing. 

For running the neural networks on other datasets, change the directories for the datasets in configuration.

There are 4 different neural networks implemented: MLP, LSTM, CNN1D, CNN2D. MLP, LSTM and CNN1D uses generalized cross correlation (GCC PHAT) values from 2 microphones to do speaker counting. CNN2D uses Short Time Fourier Transform (STFT) to do speaker counting. 

The features of the dataloader, module, and trainer can also be changed. The recommended values are given in configuration file. 

Neural network can be created, trained, evaluated and tested using the following command:

4. `python main.py`