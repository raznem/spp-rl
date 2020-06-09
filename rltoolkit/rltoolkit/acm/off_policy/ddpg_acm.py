import numpy as np
import torch
from torch.nn import functional as F

from rltoolkit.acm.off_policy import AcMOffPolicy
from rltoolkit.algorithms import DDPG
from rltoolkit.algorithms.ddpg.models import Actor, Critic


class DDPG_AcM(AcMOffPolicy, DDPG):
    def __init__(
        self, unbiased_update: bool = False, custom_loss: float = 0.0, *args, **kwargs,
    ):
        f"""DDPG with AcM class

        Args:
            unbiased_update (bool, optional): Use next_obs as action for update.
                Defaults to { False }.
        """
        super().__init__(*args, **kwargs)
        self.unbiased_update = unbiased_update
        self.actor = Actor(
            self.ob_dim, ac_lim=self.actor_ac_lim, ac_dim=self.actor_output_dim
        )
        if not self.acm_critic:
            self.critic = Critic(self.ob_dim, ac_dim=self.actor_output_dim)

        self.custom_loss = custom_loss
        if self.custom_loss:
            self.loss["ddpg"] = 0.0
            self.loss["dist"] = 0.0

        new_hparams = {
            "hparams/unbiased_update": self.unbiased_update,
            "hparams/custom_loss": self.custom_loss,
        }
        self.hparams_acm.update(new_hparams)
        self.hparams.update(self.hparams_acm)

    def noise_action(self, obs, act_noise, deterministic=False):
        action, _ = self._actor.act(obs, deterministic)
        noise = act_noise * torch.randn(self.actor_output_dim, device=self.device)
        action += noise * self.actor_ac_lim
        action = np.clip(
            action.cpu(), -1.1 * self.actor_ac_lim.cpu(), 1.1 * self.actor_ac_lim.cpu()
        )
        action = action.to(self.device)
        if self.denormalize_actor_out:
            action = self.replay_buffer.denormalize(action)
        return action

    def acm_update_condition(self):
        return (
            self.iteration > 0
            and self.acm_epochs > 0
            and self.stats_logger.frames % self.acm_update_freq == 0
        )

    def make_unbiased_update(self):
        if self.update_condition():
            for _ in range(self.grad_steps):
                batch = self.replay_buffer.sample_batch(
                    self.update_batch_size, self.device
                )
                obs, next_obs, _, reward, done, acm_action = batch
                self.update(
                    obs=obs,
                    next_obs=next_obs,
                    action=next_obs,
                    reward=reward,
                    done=done,
                    acm_action=acm_action,
                )

    def make_update(self):
        if self.unbiased_update:
            self.make_unbiased_update()
        else:
            super().make_update()

        if self.acm_update_condition():
            if self.acm_update_batches:
                self.update_acm_batches(self.acm_update_batches)
            else:
                self.update_acm(self.acm_epochs)

    def collect_params_dict(self):
        params_dict = super().collect_params_dict()
        params_dict["acm"] = self.acm.state_dict()
        return params_dict

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.acm.load_state_dict(params_dict["acm"])

    def save_model(self, save_path=None):
        save_path = DDPG.save_model(self, save_path)
        torch.save(self.acm.state_dict(), save_path + "_acm_model.pt")

    def compute_qfunc_targ(
        self, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor
    ):
        """Compute targets for Q-functions

        Args:
            reward (torch.Tensor): batch of rewards
            next_obs (torch.Tensor): batch of next observations
            done (torch.Tensor): batch of done

        Returns:
            torch.Tensor: Q-function targets for the batch
        """
        with torch.no_grad():
            next_action, _ = self.actor_targ(next_obs)
            next_action = self.replay_buffer.denormalize(next_action)
            if self.acm_critic:
                acm_obs = torch.cat([next_obs, next_action], axis=1)
                next_action = self.acm(acm_obs)
            q_target = self.critic_targ(next_obs, next_action)

            qfunc_target = reward + self.gamma * (1 - done) * q_target

        return qfunc_target

    def compute_pi_loss(self, obs, next_obs):
        action, _ = self._actor(obs)
        denorm_action = self.replay_buffer.denormalize(action)
        if self.acm_critic:
            acm_obs = torch.cat([obs, denorm_action], axis=1)
            critic_action = self.acm(acm_obs)
        else:
            critic_action = denorm_action
        loss = -self._critic(obs, critic_action).mean()

        if self.custom_loss:
            self.loss["ddpg"] = loss.item()
            if self.norm_closs:
                next_obs = self.replay_buffer.normalize(next_obs, force=True)
            else:
                action = denorm_action
            loss_dist = F.mse_loss(action, next_obs)
            self.loss["dist"] = loss_dist.item()
            loss += self.custom_loss * loss_dist

        return loss

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        acm_action: torch.Tensor,
    ):
        """DDPG update step

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
            acm_action (torch.Tensor): tensor of acm actions
        """
        for param in self.acm.parameters():
            param.requires_grad = False

        if self.acm_critic:
            action = acm_action

        y = self.compute_qfunc_targ(reward, next_obs, done)

        # Update Q-function by one step
        y_q = self._critic(obs, action)
        loss_q = F.mse_loss(y_q, y)

        self.loss["critic"] = loss_q.item()

        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        # Update policy by one step
        self._critic.eval()

        loss = self.compute_pi_loss(obs, next_obs)
        self.loss["actor"] = loss.item()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Update target networks

        self.update_target_nets()

        self._critic.train()

        for param in self.acm.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    with torch.cuda.device(0):
        model = DDPG_AcM(
            # unbiased_update=True,
            # custom_loss=True,
            # acm_update_batches=50,
            # denormalize_actor_out=True,
            env_name="Pendulum-v0",
            buffer_size=50000,
            act_noise=0.05,
            iterations=100,
            gamma=0.99,
            batch_size=200,
            stats_freq=5,
            test_episodes=3,
            # tensorboard_dir="logs_ddpg",
            # tensorboard_comment="",
            acm_update_freq=200,
            acm_epochs=1,
            acm_pre_train_epochs=10,
            acm_pre_train_samples=10000,
            use_gpu=True,
            render=False,
        )
        model.pre_train()
        model.train()
