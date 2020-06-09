import os
import tempfile

from rltoolkit.acm.on_policy import A2C_AcM, PPO_AcM

ENV_NAME = "Pendulum-v0"
ACM_PRE_TRAIN_SAMPLES = 100
ACM_PRE_TRAIN_EPOCHS = 2
ITERATIONS = 10
BATCH_SIZE = 200
STATS_FREQ = 5
ACM_UPDATE_FREQ = 200
ACM_EPOCHS = 1


def test_a2c_acm():
    model = A2C_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
    )
    model.pre_train()
    model.train()


def test_a2c_acm_no_buffer():
    model = A2C_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=None,
    )
    model.pre_train()
    model.train()


def test_ppo_acm():
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_min_max():
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
        min_max_denormalize=True,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_custom_loss_min_max():
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
        min_max_denormalize=True,
        custom_loss=0.1,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_drop_pretrain():
    acm_keep_pretrain = False
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
        acm_keep_pretrain=acm_keep_pretrain,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_batches():
    acm_update_batches = 50
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_update_batches=acm_update_batches,
        acm_val_buffer_size=100,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_closs():
    acm_update_batches = 50
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_update_batches=acm_update_batches,
        custom_loss=1,
    )
    model.pre_train()
    model.train()


def test_ppo_acm_closs_unnormlized():
    acm_update_batches = 50
    model = PPO_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_update_batches=acm_update_batches,
        custom_loss=1,
        norm_closs=False,
    )
    model.pre_train()
    model.train()


def test_save_and_load():
    iterations = 3
    stats_freq = 3
    return_done = -1000
    tmpdir = tempfile.TemporaryDirectory()
    save_dir = os.path.join(tmpdir.name, "test_model" + ".pkl")
    model = PPO_AcM(
        env_name=ENV_NAME,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
    )
    model.pre_train()
    model.train()
    model.obs_mean = -100
    model.obs_std = 100
    model.save(path=save_dir)

    loaded_model = PPO_AcM(
        env_name=ENV_NAME,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=100,
    )
    loaded_model.load(save_dir)
    assert model.obs_mean == -100 and model.obs_std == 100

    tmpdir.cleanup()
