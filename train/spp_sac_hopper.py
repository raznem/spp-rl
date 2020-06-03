import multiprocessing as mp
from itertools import product

from rltoolkit import SAC_AcM, EvalsWrapperACM


ALGO = SAC_AcM
ENV_NAME = "Hopper-v2"
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
TENSORBOARD_DIR = "logs_hopper"
TENSORBOARD_COMMENT = "1m"
LOG_ALL = True
LOG_DIR = "logs_hopper/basic_logs"
N_CORES = 24
USE_GPU = False
### ACM ###
ACM_EPOCHS = 1
ACM_BATCH_SIZE = 100
ACM_PRE_TRAIN_SAMPLES = [10000]
ACM_PRE_TRAIN_EPOCHS = 10
ACM_UPDATE_FREQ = [1000] * 10
ACM_LR = [1e-3]
ACM_UPDATE_BATCHES = [100]
CUSTOM_LOSS = [0.2]
NORM_CLOSS = [False]
ACM_CRITIC = [True]
DENORMALIZE_ACTOR_OUT = True
MIN_MAX_DENORMALIZE = True


combinations = product(
    ACM_PRE_TRAIN_SAMPLES,
    ACM_UPDATE_FREQ,
    ACM_LR,
    ACM_UPDATE_BATCHES,
    CUSTOM_LOSS,
    NORM_CLOSS,
    ACM_CRITIC,
)


def run_combination(*args, **kwargs):
    evals = EvalsWrapperACM(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    kwargs_list = []
    for (
        acm_pre_train_samples,
        acm_update_freq,
        acm_lr,
        acm_update_batches,
        custom_loss,
        norm_closs,
        acm_critic,
    ) in combinations:
        kwargs = {
            "Algo": ALGO,
            "env_name": ENV_NAME,
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
            ### ACM ###
            "acm_epochs": ACM_EPOCHS,
            "acm_batch_size": ACM_BATCH_SIZE,
            "acm_pre_train_samples": acm_pre_train_samples,
            "acm_pre_train_epochs": ACM_PRE_TRAIN_EPOCHS,
            "acm_update_freq": acm_update_freq,
            "acm_lr": acm_lr,
            "acm_update_batches": acm_update_batches,
            "custom_loss": custom_loss,
            "denormalize_actor_out": DENORMALIZE_ACTOR_OUT,
            "min_max_denormalize": MIN_MAX_DENORMALIZE,
            "norm_closs": norm_closs,
            "acm_critic": acm_critic,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
