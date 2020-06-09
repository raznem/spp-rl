from rltoolkit.acm.off_policy import DDPG_AcM, SAC_AcM

ENV_NAME = "Pendulum-v0"
ACM_PRE_TRAIN_SAMPLES = 100
ACM_PRE_TRAIN_EPOCHS = 1
ITERATIONS = 5
BATCH_SIZE = 100
STATS_FREQ = 5
ACM_UPDATE_FREQ = 100
ACM_EPOCHS = 1
ACM_VAL_BUFFER_SIZE = 100


def test_ddpg_acm():
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=ACM_VAL_BUFFER_SIZE,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_min_max():
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=ACM_VAL_BUFFER_SIZE,
        denormalize_actor_out=True,
        min_max_denormalize=True,
    )
    model.pre_train()
    model.train()


def test_sac_acm():
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=ACM_VAL_BUFFER_SIZE,
    )
    model.pre_train()
    model.train()


def test_sac_acm_drop_pretrain():
    acm_keep_pretrain = False
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=ACM_VAL_BUFFER_SIZE,
        acm_keep_pretrain=acm_keep_pretrain,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_batches():
    acm_update_batches = 50
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_val_buffer_size=ACM_VAL_BUFFER_SIZE,
        acm_update_batches=acm_update_batches,
    )
    model.pre_train()
    model.train()


def test_ddpg_custom_loss():
    custom_loss = 0.1
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        custom_loss=custom_loss,
    )
    model.pre_train()
    model.train()


def test_ddpg_custom_loss_min_max():
    custom_loss = 0.1
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        custom_loss=custom_loss,
        denormalize_actor_out=True,
        min_max_denormalize=True,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_critic():
    acm_critic = True
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_critic=acm_critic,
    )
    model.pre_train()
    model.train()


def test_sac_acm_critic():
    acm_critic = True
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_critic=acm_critic,
    )
    model.pre_train()
    model.train()
