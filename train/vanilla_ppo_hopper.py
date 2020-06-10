import multiprocessing as mp

from rltoolkit import PPO, EvalsWrapper

ALGO = PPO
EVALS = 1
ITERATIONS = 1001
MAX_FRAMES = 1e6
RETURN_DONE = 1e4
GAMMA = 0.99
A_LR = 3e-4
C_LR = 1e-3

TENSORBOARD_DIR = "logs"
LOG_ALL = True
LOG_DIR = "logs/basic_logs"
N_CORES = 2
ENV_NAME = ["Hopper-v2"]
ENTROPY_COEF = 0.0
MAX_PPO_EPOCHS = 10
BATCH_SIZE = 2000
NORM_ALPHA = 0.95
KL_DIV_THRESHOLD = 0.1
PPO_BATCH_SIZE = 512
TEST_EPISODES = 10


def run_combination(*args, **kwargs):
    evals = EvalsWrapper(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    kwargs_list = []
    for env_name in ENV_NAME:
        kwargs = {
            "Algo": ALGO,
            "evals": EVALS,
            "iterations": ITERATIONS,
            "gamma": GAMMA,
            "actor_lr": A_LR,
            "critic_lr": C_LR,
            "batch_size": BATCH_SIZE,
            "tensorboard_dir": TENSORBOARD_DIR,
            "log_all": LOG_ALL,
            "return_done": RETURN_DONE,
            "log_dir": LOG_DIR,
            "verbose": 0,
            "render": False,
            "ppo_batch_size": PPO_BATCH_SIZE,
            "kl_div_threshold": KL_DIV_THRESHOLD,
            "env_name": env_name,
            "max_frames": MAX_FRAMES,
            "entropy_coef": ENTROPY_COEF,
            "tensorboard_comment": "",
            "max_ppo_epochs": MAX_PPO_EPOCHS,
            "obs_norm_alpha": NORM_ALPHA,
            "test_episodes": TEST_EPISODES,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
