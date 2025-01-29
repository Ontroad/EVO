from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config



class EVOOnPolicyAdapter(OnPolicyAdapter):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        self._reset_log()
        self.rollret = []  
        self.rollcost = [] 
        
        self.temp_obs = [[] for _ in range(int(self._env.num_envs))]  
        self.temp_act = [[] for _ in range(int(self._env.num_envs))]  
        self.temp_logp = [[] for _ in range(int(self._env.num_envs))] 
        self.cost_obs_act_logp = {'cost': [], 'obs': [], 'act': [], 'logp': []}  
        self.start_idx = 0  

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            for idx in range(self._env.num_envs):  
                self.temp_obs[idx+self.start_idx].append(obs[idx])  
                self.temp_act[idx+self.start_idx].append(act[idx])
                self.temp_logp[idx+self.start_idx].append(logp[idx])
            
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )
                    
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self.rollret.append(self._ep_ret[idx].item()) 
                        self.rollcost.append(self._ep_cost[idx].item())  
                        self.start_idx += 1 
                        self.temp_obs.append([])
                        self.temp_act.append([])
                        self.temp_logp.append([])
                        self.cost_obs_act_logp['cost'].append(self._ep_cost[idx].tolist())

                        self._log_metrics(logger, idx)
                        self._reset_log(idx)
                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

        self.cost_obs_act_logp['obs'] = self.temp_obs
        self.cost_obs_act_logp['act'] = self.temp_act
        self.cost_obs_act_logp['logp'] = self.temp_logp




    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
