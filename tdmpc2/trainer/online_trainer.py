from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
import gym


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(self.env.num_actions,), dtype=np.float32
		)

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=False) # have to disable for now since Genesis rendering is difficult
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action.unsqueeze(0))
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward.cpu())
			ep_successes.append(info['episode']['rew_tracking_lin_vel'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			full_reward=np.nanmean(ep_rewards),
			#episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, tuple):
			obs = obs[0]
		else:
			obs = obs
		if action is None:
			action = torch.full_like(torch.from_numpy(self.action_space.sample().astype(np.float32)).to('mps').unsqueeze(0), float(0))
		if reward is None:
			reward = torch.tensor(float(0)).to('mps').unsqueeze(0)
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
				self.logger.save_agent(self.agent, identifier=f'{self._step}')

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						full_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						rew_tracking_lin_vel=info['episode']['rew_tracking_lin_vel'],
						rew_tracking_ang_vel=info['episode']['rew_tracking_ang_vel'],
						rew_lin_vel_z=info['episode']['rew_lin_vel_z'],
						rew_base_height=info['episode']['rew_base_height'],
						rew_action_rate=info['episode']['rew_action_rate'],
						rew_similar_to_default=info['episode']['rew_similar_to_default'],
						rew_feet_air_time=info['episode']['rew_feet_air_time'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1).to('mps').unsqueeze(0)
			else:
				action = torch.from_numpy(self.action_space.sample().astype(np.float32)).to('mps').unsqueeze(0)
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
