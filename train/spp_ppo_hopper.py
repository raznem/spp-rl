import multiprocessing as mp
from itertools import product

import rltoolkit as rl

ENV_NAME = "Hopper-v2"
NAME = ""

TENSORBOARD_DIR = NAME + "logs"
LOG_DIR = NAME + "logs/basic_logs"
RETURN_DONE = 5e4
CORES = 80

EVALS = 3
ITERATIONS = 1001
MAX_FRAMES = 1e6
BATCH_SIZE = [2000]  # 3
GAMMA = [0.99]  # 6
A_LR = 3e-4
C_LR = 3e-4
NORM_ALPHA = 0.99
PPO_MAX_KL_DIV = 0.1
MAX_PPO_EPOCHS = 10
PPO_BATCH_SIZE = [512]  # 12
ENTROPY_COEF = 0

ACM_EPOCHS = [5]  # 24
ACM_BATCH_SIZE = [64]  # 48
ACM_UPDATE_FREQ = [3]  # 96
ACM_LR = [3e-4]  # 96
ACM_PRE_TRAIN_SAMPLES = 1e5
PRE_TRAIN_EPOCHS = 5
DENORMALIZE_ACTOR_OUT = True  #
MIN_MAX_DENORMALIZE = True  #
CUSTOM_LOSS = [0.1]  # 4 * 96 * 3(evals)

TEST_EPISODES = 10


def run_combination(*args, **kwargs):
    evals = rl.EvalsWrapperACM(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    kwargs_list = []
    combinations = product(
        BATCH_SIZE,
        PPO_BATCH_SIZE,
        ACM_EPOCHS,
        ACM_BATCH_SIZE,
        ACM_UPDATE_FREQ,
        CUSTOM_LOSS,
        ACM_LR,
        GAMMA,
    )
    for (
        batch_size,
        ppo_batch_size,
        acm_epochs,
        acm_batch_size,
        acm_update_freq,
        custom_loss,
        acm_lr,
        gamma,
    ) in combinations:
        kwargs = {
            "Algo": rl.PPO_AcM,
            "iterations": ITERATIONS,
            "gamma": gamma,
            "actor_lr": A_LR,
            "critic_lr": C_LR,
            "batch_size": batch_size,
            "tensorboard_dir": TENSORBOARD_DIR,
            "return_done": RETURN_DONE,
            "log_dir": LOG_DIR,
            "verbose": 0,
            "render": False,
            "ppo_batch_size": ppo_batch_size,
            "kl_div_threshold": PPO_MAX_KL_DIV,
            "env_name": ENV_NAME,
            "max_frames": int(MAX_FRAMES),
            "entropy_coef": ENTROPY_COEF,
            "log_all": True,
            "max_ppo_epochs": MAX_PPO_EPOCHS,
            "obs_norm_alpha": NORM_ALPHA,
            "denormalize_actor_out": DENORMALIZE_ACTOR_OUT,
            "acm_epochs": acm_epochs,
            "acm_batch_size": acm_batch_size,
            "acm_update_freq": acm_update_freq,
            "acm_lr": acm_lr,
            "acm_pre_train_samples": ACM_PRE_TRAIN_SAMPLES,
            "acm_pre_train_epochs": PRE_TRAIN_EPOCHS,
            "test_episodes": TEST_EPISODES,
            "evals": EVALS,
            "min_max_denormalize": MIN_MAX_DENORMALIZE,
            "custom_loss": custom_loss,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
