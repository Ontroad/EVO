from __future__ import annotations

import torch

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer


class EVTOnPolicyBuffer(OnPolicyBuffer):  # pylint: disable=too-many-instance-attributes
    
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
        )

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
    ) -> None:
        
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
       
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data


class EVTVectorOnPolicyBuffer(VectorOnPolicyBuffer):
    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOnPolicyBuffer`."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c

        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers: list[EVTOnPolicyBuffer] = [
            EVTOnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """Get the data in the buffer.

        In vector-on-policy buffer, we get the data from each buffer and then concatenate them.
        """
        self.buffers[idx].finish_path(last_value_r, last_value_c)

