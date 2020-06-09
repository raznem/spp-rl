import logging

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rltoolkit import utils
from rltoolkit.acm import AcMTrainer
from rltoolkit.algorithms import A2C, PPO
from rltoolkit.algorithms.ppo.advantage_dataset import AcMAdvantageDataset
from rltoolkit.basic_model import Actor, Critic
from rltoolkit.buffer import Memory, MemoryAcM

logger = logging.getLogger(__name__)


class AcMOnPolicyTrainer(AcMTrainer):
    def __init__(self, custom_loss: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalized_buffer = self.denormalize_actor_out

        self.custom_loss = custom_loss
        self.actor = Actor(
            self.ob_dim, self.actor_ac_lim, self.actor_output_dim, discrete=False
        )
        self.critic = Critic(self.ob_dim)
        if self.custom_loss:
            self.loss["policy"] = 0.0
            self.loss["dist"] = 0.0

        new_hparams = {"hparams/custom_loss": self.custom_loss}
        self.hparams_acm.update(new_hparams)

    def process_action(self, action: torch.Tensor, obs: torch.tensor, *args, **kwargs):
        """Pre-processing of action before it will go the env.
        It will not be saved to the buffer.

        Args:
            action (torch.Tensor): action from the policy
            obs (torch.tensor): observations for this actions

        Returns:
            [np.array]: processed action
        """
        with torch.no_grad():
            if self.denormalize_actor_out:
                action = self.replay_buffer.denormalize(action)
            acm_features = torch.cat((obs, action), axis=1)
            acm_action = self.acm.act(acm_features)
            acm_action = acm_action.cpu().numpy()[0]
            self.buffer.add_acm_action(acm_action)

        return acm_action

    @utils.measure_time
    def perform_iteration(self):
        """Single train step of algorithm

        Returns:
            Memory: Buffer filled with one batch
            float: Time taken for evaluation
        """
        self.buffer = MemoryAcM(
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            device=self.device,
            alpha=self.obs_norm_alpha,
            max_obs=self.max_obs,
            min_obs=self.min_obs,
            min_max_denormalize=self.min_max_denormalize,
        )
        self.collect_batch(self.buffer)

        advantages = self.update_critic(self.buffer)
        self.update_actor(advantages, self.buffer)

        self.replay_buffer.add_buffer(self.buffer)
        if self.acm_update_freq and self.iteration % self.acm_update_freq == 0:
            if self.acm_update_batches:
                self.update_acm_batches(self.acm_update_batches)
            else:
                self.update_acm(self.acm_epochs)

        if self.denormalize_actor_out:
            self.replay_buffer = self.update_obs_mean_std(self.replay_buffer)
        return self.buffer

    def update_actor(self, advantages: torch.Tensor, buffer: Memory):
        """One iteration of actor update

        Args:
            advantages (torch.Tensor): advantages for observations from buffer
            buffer (Memory): buffer with samples
        """
        if self.custom_loss:
            self.update_actor_acm(advantages, buffer)
        else:
            super().update_actor(advantages, buffer)

    def update_actor_acm(self, advantages: torch.Tensor, buffer: Memory):
        if self.normalize_adv:
            advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-8
            )

        actions = torch.cat(buffer.actions)
        next_obs = buffer.next_obs

        if self.norm_closs:
            next_obs = buffer.normalize(next_obs, force=True)
        else:
            actions = buffer.denormalize(actions)

        action_logprobs = buffer.action_logprobs
        action_logprobs = torch.cat(action_logprobs)

        actor_loss = (-action_logprobs * advantages).mean()
        self.loss["actor"] = actor_loss.item()
        dist_loss = F.mse_loss(actions, next_obs)
        self.loss["dist"] = dist_loss.item()
        policy_loss = actor_loss + self.custom_loss * dist_loss
        policy_loss.backward()
        self.actor_optimizer.step()
        self.loss["policy"] = policy_loss.item()

    def add_tensorboard_logs(self, buffer: MemoryAcM, done: bool):
        super().add_tensorboard_logs(buffer, done)
        if self.debug_mode:
            self.tensorboard_writer.plot_dist_loss(
                self.iteration, buffer, on_policy=True
            )


class A2C_AcM(AcMOnPolicyTrainer, A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.update(self.hparams_acm)
        if self.denormalize_actor_out and self.obs_norm_alpha:
            logger.warning("In AcM normalization obs_norm_alpha is redundant.")
        elif self.obs_norm_alpha:
            logger.warning(
                "You are using redundant variable obs_norm_alpha in the AcM."
            )

    def save_model(self, save_path=None):
        save_path = A2C.save_model(self, save_path)
        torch.save(self.acm.state_dict(), save_path + "_acm_model.pt")

    def collect_params_dict(self):
        params_dict = super().collect_params_dict()
        params_dict["acm"] = self.acm.state_dict()
        return params_dict

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.acm.load_state_dict(params_dict["acm"])


class PPO_AcM(A2C_AcM, PPO):
    def save_model(self, save_path=None):
        save_path = PPO.save_model(self, save_path)
        torch.save(self.acm.state_dict(), save_path + "_acm_model.pt")

    def update_actor_acm(self, advantages: torch.Tensor, buffer: Memory):
        advantage_dataset = AcMAdvantageDataset(advantages, buffer, self.normalize_adv)
        dataloader = DataLoader(
            advantage_dataset, batch_size=self.ppo_batch_size, shuffle=True
        )

        kl_div = 0.0
        self.loss["actor"] = 0
        self.loss["entropy"] = 0
        self.loss["policy"] = 0
        self.loss["dist"] = 0

        for i in range(self.max_ppo_epochs):
            if kl_div >= self.kl_div_threshold:
                break

            for (
                advantages,
                action_logprobs,
                actions,
                norm_obs,
                next_obs,  # it is normalized next obs
            ) in dataloader:
                if not self.norm_closs:
                    next_obs = buffer.denormalize(next_obs)
                    actions = buffer.denormalize(actions)
                new_dist = self.actor.get_actions_dist(norm_obs)
                new_logprobs = new_dist.log_prob(torch.squeeze(actions))

                entropy = new_dist.entropy().mean()

                actor_loss = self._clip_loss(action_logprobs, new_logprobs, advantages)
                ppo_loss = actor_loss - self.entropy_coef * entropy

                dist_loss = F.mse_loss(actions, next_obs)

                policy_loss = ppo_loss + self.custom_loss * dist_loss
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()
                self.loss["actor"] += actor_loss.item()
                self.loss["entropy"] += entropy.item()
                self.loss["policy"] += policy_loss.item()
                self.loss["dist"] += dist_loss.item()

            kl_div = utils.kl_divergence(action_logprobs.cpu(), new_logprobs.cpu())

        self.loss["actor"] /= i + 1
        self.loss["entropy"] /= i + 1
        self.loss["policy"] /= i + 1
        self.loss["dist"] /= i + 1
        logger.debug(f"PPO update finished after {i} epochs with KL = {kl_div}")
        self.kl_div_updates_counter += i + 1


if __name__ == "__main__":
    torch.set_num_threads(1)
    model = PPO_AcM(
        env_name="HalfCheetah-v2",
        gamma=0.99,
        acm_pre_train_samples=10000,
        acm_pre_train_epochs=3,
        iterations=1000,
        batch_size=1000,
        stats_freq=5,
        acm_update_freq=5,
        acm_epochs=1,
        acm_lr=1e-4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        kl_div_threshold=0.1,
        max_ppo_epochs=10,
        ppo_batch_size=512,
        acm_batch_size=64,
        denormalize_actor_out=True,
        min_max_denormalize=True,
        custom_loss=0.5,
        tensorboard_dir="tmp_logs",
        # acm_update_batches=50,
        obs_norm=True,
        test_episodes=3,
    )
    model.pre_train()
    model.train()
