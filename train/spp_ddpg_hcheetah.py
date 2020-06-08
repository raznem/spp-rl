from itertools import product

from rltoolkit import  EvalsWrapperACM, EvalsWrapper,  DDPG_AcM, DDPG, SAC_AcM
from rltoolkit.acm.models.basic_acm import BasicAcM

import multiprocessing as mp

acm = BasicAcM(34 , 6 ,  False)

ENV_NAME = "HalfCheetah-v2"
ALGO = DDPG_AcM
EVALS = 10
MAX_FRAMES = 1000000

LOG_ALL = True
TEST_EPISODES = 10
UBS = 100
STATS_FREQ = 1

RANDOM_FRAMES = 0
GAMMA = [ 0.95 ]
BATCH_SIZE = 5000
DDPG_LR = 5e-4


RUNID = 1
TENSORBOARD_DIR = "%s_final_run%d" % (ENV_NAME, RUNID)
LOG_DIR = "%s_final_run%d/basic_logs" % (ENV_NAME, RUNID)
N_CORES = 30


ACM_EPOCHS = 1
ACM_UPDATE_FREQ = 500 
ACM_PRE_TRAIN_SAMPLES = 20000
ACM_PRE_TRAIN_EPOCHS = 5
ACM_LR = 0.005
ACM_BATCH = 128
ACT_NOISE = 0.05

CUSTOM_LOSS = 1.
NORM_CLOSS = False 

ACM_UPDATE_BATCHES = 200 

def run_combination(*args, **kwargs):
    evals = EvalsWrapperACM(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    kwargs_list = []
    for  gamma in GAMMA:
        for i in range(EVALS):
            kwargs = {
                "Algo": ALGO,
                "acm_model": acm,
                "act_noise" : ACT_NOISE,
                "evals" : 1,
                "actor_lr" : DDPG_LR,
                "critic_lr" : DDPG_LR,
                "obs_norm" : False,
                "env_name": ENV_NAME, 
                "custom_loss" : CUSTOM_LOSS,
                "iterations": MAX_FRAMES / BATCH_SIZE,
                "buffer_size": MAX_FRAMES,
                "unbiased_update" : False,
                "gamma": gamma,
                "update_batch_size": UBS,
                "test_episodes": TEST_EPISODES,
                "batch_size": BATCH_SIZE,
                "acm_batch_size": ACM_BATCH,
                "tensorboard_dir": TENSORBOARD_DIR,
                "log_dir": LOG_DIR,
                "max_frames": MAX_FRAMES,
                "stats_freq": STATS_FREQ,
                "random_frames": 0,
                "log_all": LOG_ALL,
                "acm_epochs": 1,
                "acm_update_freq": ACM_UPDATE_FREQ,
                "acm_pre_train_epochs": 5,
                "acm_pre_train_samples": ACM_PRE_TRAIN_SAMPLES,
                "acm_update_batches" : ACM_UPDATE_BATCHES,
                "acm_lr" : ACM_LR,
                "verbose": 1,
                "render": False,
                "min_max_denormalize" : True,
                "denormalize_actor_out" : True,
                "acm_critic" : True,
                "norm_closs" : NORM_CLOSS       ,
                "debug_mode" : True,
                "tensorboard_comment" : "closs%s_actrue_run%d" % (NORM_CLOSS, i),
            }
            kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)