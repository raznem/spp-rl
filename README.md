# Software and Results for the Paper Entitled  *State Planning Policy Reinforcement Learning*

This repository is the official implementation of the State Planning Policy Reinforcement Learning.  
Demo [video](https://youtu.be/dWnhNnX6f0g).

<img src="plots/spprl.jpg" alt="SPPRL" width="500"/>

## Requirements

Code was run on Ubuntu 18.03 in anaconda environment, in case of another set-up, extra dependencies could be required.
To install requirements run:

```setup
pip install -r rltoolkit/requirements.txt
```

Then install `rltoolkit` with:
```rltoolkit install
pip install -e rltoolkit/
```

## Training

To train the models in the paper, you can use scripts from `train` folder.
For example, to train SPP-SAC on the hopper, simply run:

```train
python train/spp_sac_hopper.py
```

take note of the `N_CORES` parameter within the training scripts, which 
should be set accordingly to the available CPU unit(s).

## Evaluation

Model evaluation code is available in the jupyter notebook: `notebooks/load_and_test.ipynb`.
There you can load pre-trained models, evaluate their reward, and render in the environment.


## Pre-trained Models

You can find pre-trained models in `models` directory and check how to load them in `load_and_test.ipynb` notebook.


## Results

Our model achieves the following performance on :

### [OpenAI gym MuJoCo environments](https://gym.openai.com/envs/#mujoco)

# TODO: plots with performance on different envs.
