import torch

from rltoolkit import config
from rltoolkit.acm import AcMTrainer
from rltoolkit.buffer import BufferAcMOffPolicy, MemoryAcM


class AcMOffPolicy(AcMTrainer):
    def __init__(
        self,
        buffer_size: int = config.BUFFER_SIZE,
        acm_critic: bool = config.ACM_CRITIC,
        *args,
        **kwargs,
    ):
        f"""Off policy acm meta class

        Args:
            buffer_size (int, optional): Size of the replay buffer.
                Defautls to { config.BUFFER_SIZE }.
            acm_critic (bool): use actions of acm for critic.
                Defaults to { config.ACM_CRITIC }.
        """
        super().__init__(*args, **kwargs)

        self.buffer_size = buffer_size
        self.acm_critic = acm_critic
        self.replay_buffer = BufferAcMOffPolicy(
            self.buffer_size,
            self.ob_dim,
            self.actor_output_dim,
            acm_act_shape=self.ac_dim,
            acm_discrete=self.discrete,
            dtype=torch.float32,
            device=self.device,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            min_max_denormalize=self.min_max_denormalize,
            max_obs=self.max_obs,
            min_obs=self.min_obs,
            obs_norm=self.obs_norm,
        )
        self.max_ep_len = None
        new_hparams = {
            "hparams/buffer_size": self.buffer_size,
            "hparams/acm_critic": self.acm_critic,
        }
        self.hparams_acm.update(new_hparams)

    def initial_act(self, obs) -> torch.Tensor:
        action = self.actor_ac_lim * torch.randn(1, self.ob_dim, device=self.device)
        if self.denormalize_actor_out:
            action = self.replay_buffer.denormalize(action)
        return action

    def collect_samples(self):
        """Collect samples into buffer for the pre-train stage
            In contrast do DDPG loop here we are adding next_obs instead of actions.
        """
        # TODO refactor this to not to duplicate collect from DDPG
        #      - not so easy due to logger :(
        collected = 0
        while collected < self.acm_pre_train_samples:
            obs = self.env.reset()
            end = False
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.unsqueeze(0)

            prev_idx = self.replay_buffer.add_obs(obs)
            ep_len = 0

            while not end:
                acm_action = AcMTrainer.initial_act(self, obs)
                self.replay_buffer.add_acm_action(acm_action)
                obs, rew, done, _ = self.env.step(acm_action)
                ep_len += 1

                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                obs = obs.unsqueeze(0)

                end = done
                done = False if ep_len == self.max_ep_len else done

                next_idx = self.replay_buffer.add_obs(obs)
                self.replay_buffer.add_timestep(prev_idx, next_idx, obs, rew, done, end)
                prev_idx = next_idx
                collected += 1

    def process_action(
        self, action: torch.Tensor, obs: torch.tensor, pre_train: bool = False
    ):
        """Pre-processing of action before it will go the env.

        Args:
            action (torch.Tensor): action from the policy.
            obs (torch.tensor): observations for this actions.

        Returns:
            np.array: processed action
        """
        with torch.no_grad():
            acm_observation = torch.cat([obs, action], axis=1)
            acm_action = self.acm.act(acm_observation)
            acm_action = acm_action.cpu().numpy()[0]
            self.replay_buffer.add_acm_action(acm_action)
        return acm_action

    def add_tensorboard_logs(self, buffer: MemoryAcM, done: bool):
        super().add_tensorboard_logs(buffer, done)
        if self.debug_mode:
            self.tensorboard_writer.plot_dist_loss(self.iteration, buffer)
