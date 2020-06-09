from itertools import product

from rltoolkit import DDPG, EvalsWrapper
import multiprocessing as mp

import torch
import numpy

ENV_NAME = "HalfCheetah-v2"
ALGO = DDPG
EVALS = 10
ITERATIONS = 200
MAX_FRAMES = 1000000
RETURN_DONE = 10000
LOG_ALL = True
TEST_EPISODES = 10
UPDATE_BATCH_SIZE = 100
STATS_FREQ = 1

RANDOM_FRAMES = 1000
GAMMA = 0.99
BATCH_SIZE = 5000

TENSORBOARD_DIR = "%s_vanilladdpg_final" % (ENV_NAME)
LOG_DIR = "%s_vanilladdpg_final/basic_logs" % (ENV_NAME)
N_CORES = 2


kwargsarr = [{
    "env_name": ENV_NAME,
    "Algo": ALGO,
    "evals": 1,
    "iterations": ITERATIONS,
    "gamma": GAMMA,
    "update_batch_size": UPDATE_BATCH_SIZE,
    "test_episodes" : TEST_EPISODES,
    "batch_size": BATCH_SIZE,
    "tensorboard_dir": TENSORBOARD_DIR,
    "return_done": RETURN_DONE,
    "log_dir": LOG_DIR,
    "max_frames": MAX_FRAMES,
    "log_all": LOG_ALL,
    "return_done": RETURN_DONE,
    "stats_freq" : STATS_FREQ,
    "random_frames" : RANDOM_FRAMES,
    "tensorboard_comment" : "run%d" %(i),
    "verbose": 1,
    "render": False,
} for i in range(EVALS)]

def apply_kwargs(kwargs):
    evals = EvalsWrapper(**kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()
    

if __name__ == "__main__":
  with mp.Pool(N_CORES) as p:
        p.map(apply_kwargs, kwargsarr)
