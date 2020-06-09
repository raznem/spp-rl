from rltoolkit import A2C, DDPG, PPO_AcM, SAC_AcM

ITERATIONS = 100
MAX_FRAMES = 2e2
STATS_FREQ = 1
ENV = "Pendulum-v0"
BATCH_SIZE = 70
ACM_PRETRAIN_SAMPLES = 10
ACM_PRETRAIN_EPOCHS = 2
OFF_POLICY_BUFFER_SIZE = 100
DEBUG_MODE = True
TENSORBOARD_DIR = ""
TEST_EPISODES = 1


def test_ddpg(tmpdir):
    agent = DDPG(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.train()


def test_ddpg_no_debug(tmpdir):
    agent = DDPG(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=(not DEBUG_MODE),
        test_episodes=TEST_EPISODES,
    )
    agent.train()


def test_a2c(tmpdir):
    agent = A2C(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.train()


def test_a2c_no_norm(tmpdir):
    agent = A2C(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=DEBUG_MODE,
        obs_norm_alpha=None,
        test_episodes=TEST_EPISODES,
    )
    agent.train()


def test_ppo_acm(tmpdir):
    agent = PPO_AcM(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.pre_train()
    agent.train()


def test_ppo_acm_no_debug(tmpdir):
    agent = PPO_AcM(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=not DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.pre_train()
    agent.train()


def test_sac_acm(tmpdir):
    agent = SAC_AcM(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.pre_train()
    agent.train()


def test_sac_acm_no_debug(tmpdir):
    agent = SAC_AcM(
        tensorboard_dir=TENSORBOARD_DIR or tmpdir,
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        debug_mode=not DEBUG_MODE,
        test_episodes=TEST_EPISODES,
    )
    agent.pre_train()
    agent.train()
