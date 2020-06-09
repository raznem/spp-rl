# Software and Results for the Paper Entitled  *State Planning Policy Reinforcement Learning*

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

Code was run on Ubuntu 18 in anaconda environment, in case of other set-up extra dependencies could be required.
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
For example, to train SPP-SAC on hopper, simply run:

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

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
