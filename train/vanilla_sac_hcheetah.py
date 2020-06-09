import multiprocessing as mp
from itertools import product

from rltoolkit import SAC, EvalsWrapper


ALGO = SAC
ENV_NAME = ["HalfCheetah-v2"] * 10
EVALS = 1
ITERATIONS = 1000
MAX_FRAMES = int(1e6)
BATCH_SIZE = 1000
TEST_EPISODES = 3
STATS_FREQ = 5
GAMMA = 0.99
LEARNING_RATE = 1e-3
ALPHA_LR = 1e-3
ALPHA = 0.2
UPDATE_BATCH_SIZE = 100
RANDOM_FRAMES = 1000
UPDATE_FREQ = 50
GRAD_STEPS = 50
TENSORBOARD_DIR = "logs_sac_hcheetah"
TENSORBOARD_COMMENT = ""
LOG_ALL = True
LOG_DIR = "logs_sac_hcheetah/basic_logs"
N_CORES = 2
USE_GPU = False


combinations = product(ENV_NAME,)


def run_combination(*args, **kwargs):
    evals = EvalsWrapper(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    kwargs_list = []
    for env_name in combinations:
        kwargs = {
            "Algo": ALGO,
            "env_name": env_name,
            "evals": EVALS,
            "iterations": ITERATIONS,
            "max_frames": MAX_FRAMES,
            "batch_size": BATCH_SIZE,
            "test_episodes": TEST_EPISODES,
            "stats_freq": STATS_FREQ,
            "gamma": GAMMA,
            "actor_lr": LEARNING_RATE,
            "critic_lr": LEARNING_RATE,
            "alpha_lr": ALPHA_LR,
            "alpha": ALPHA,
            "update_batch_size": UPDATE_BATCH_SIZE,
            "random_frames": RANDOM_FRAMES,
            "update_freq": UPDATE_FREQ,
            "grad_steps": GRAD_STEPS,
            "tensorboard_dir": TENSORBOARD_DIR,
            "tensorboard_comment": TENSORBOARD_COMMENT,
            "log_all": LOG_ALL,
            "log_dir": LOG_DIR,
            "use_gpu": USE_GPU,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
