# Usage of the Python Files
This README contains an explanation of what the Python files do.

## environment.py
This Python file sets up the simulation environment with the trucks, the scene, and the dummy trucks. It is also the 
file that gets observation from Vortex and also the file that moves the truck in the simulation. It also defines the 
reward function. 

## td3_train.py
This Python file imports the Model from td3_model.py and the Environment from environment.py and trains the TD3 model. 
It also plots and saves the learning curve, and the reward distribution plot. It also prints the model summaries and 
puts the current hyperparameters into a log file. It also saves recordings of the run every so often. The model, 
recording, log file and plots are all saved in a folder under the td3_models directory. 

This is the file we run to train the RL models.

## variables.py
This Python file contains all the variables that we may want to change when training different models. It
contains the hyperparameters we might need to change. It is this file that we need to change to train different models.

## play_recordings.py
This Python file plays the Vortex recordings. To view recordings, enter the full path of the recording you wish to
watch in the "recording_file_location" variable.

## ddpg.py
This Python file used to train the DDPG model. However, as the ddpg algorithm could not learn to park our truck, we 
moved on to TD3. This file has not been updated to match our modified environment and variables file; thus, it most likely
deprecated.