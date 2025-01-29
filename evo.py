
from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genpareto
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from omnisafe.algorithms.on_policy.second_order import CPO
from omnisafe.algorithms.on_policy.evt.evt_rollout import EVOOnPolicyAdapter
from omnisafe.algorithms.on_policy.evt.evt_buffer import EVTVectorOnPolicyBuffer
from omnisafe.common.buffer import VectorOnPolicyBuffer
from torch.utils.data import DataLoader, TensorDataset
import time


@registry.register
class EVO(CPO):

    def _init_env(self) -> None:
        self._env: EVOOnPolicyAdapter = EVOOnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        self._buf: EVTVectorOnPolicyBuffer = EVTVectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )
        self.use_cost_buffer = self._cfgs.algo_cfgs.use_cost_buffer
        self.use_reward_buffer = self._cfgs.algo_cfgs.use_reward_buffer
        self.buffer_size = self._cfgs.algo_cfgs.cost_buffer_size
        self.reward_buffer_size = self._cfgs.algo_cfgs.reward_buffer_size
        self.distribution_percentile = self._cfgs.algo_cfgs.initial_distribution_percentile 
        self.distribution_reward_percentile = self._cfgs.algo_cfgs.initial_distribution_reward_percentile  
        
        self.epoch_reward = []  
        if self.use_cost_buffer:
            self.cost_buffer_cost = []  
            self.cost_buffer_obs = []
            self.cost_buffer_act = []
            self.cost_buffer_logp = []
        if self.use_reward_buffer:
            self.advr_obs_act_logp = {'advr': [], 'obs': [], 'act': [], 'logp': []}  
            self.reward_buffer = [] 

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/threshold')

    def _update(self) -> None:
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        rollobs = self._env.temp_obs 
        rollact = self._env.temp_act
        rolllogp = self._env.temp_logp

        
        if self.use_reward_buffer:
            
            self.epoch_reward.append(np.mean(self._env.rollret))
            for i in range(obs.shape[0]):  
                if len(self.advr_obs_act_logp['advr']) >= self.reward_buffer_size:
                    self.advr_obs_act_logp['advr'].pop(0)
                    self.advr_obs_act_logp['obs'].pop(0)
                    self.advr_obs_act_logp['act'].pop(0)
                    self.advr_obs_act_logp['logp'].pop(0)
                self.advr_obs_act_logp['advr'].append(adv_r[i].cpu().numpy())   
                self.advr_obs_act_logp['obs'].append(obs[i].cpu().numpy())
                self.advr_obs_act_logp['act'].append(act[i].cpu().numpy())
                self.advr_obs_act_logp['logp'].append(logp[i].cpu().numpy())
            
            reward_buffer_obs = torch.stack([torch.tensor(obs) for obs in self.advr_obs_act_logp['obs']]).to(self._device) 
            reward_buffer_act = torch.stack([torch.tensor(act) for act in self.advr_obs_act_logp['act']]).to(self._device)
            reward_buffer_logp = torch.stack([torch.tensor(logp) for logp in self.advr_obs_act_logp['logp']]).to(self._device)
            reward_buffer_advr = torch.stack([torch.tensor(advr) for advr in self.advr_obs_act_logp['advr']]).to(self._device)
            
            logp_reward_current = []
            start = time.time()
            with torch.no_grad():
                tmp_dist = self._actor_critic.actor(reward_buffer_obs)

                logp_reward_current = tmp_dist.log_prob(reward_buffer_act).sum(axis=-1)  
            end = time.time()
            
            logp_diff = (logp_reward_current - reward_buffer_logp) 
            
            ratio = torch.exp(logp_diff)/torch.exp(logp_diff).sum()
           
            reward_ratio_list = [(reward_buffer_advr[i], ratio[i]) for i in range(len(reward_buffer_advr))]
            

        
        self.distribution_reward_percentile = np.clip(self.distribution_reward_percentile, 0.2, 0.9)
        
    
        adv_r1 = reward_buffer_advr.cpu().numpy()  
        if (np.all(adv_r1 == 0)):
            threshold0 = 0
        else:
            start = time.time()
            if len(reward_ratio_list) > 0:
                weighted_samples = []
                
                advr_array = torch.stack([x[0] for x in reward_ratio_list]).cpu().numpy() 
                ratio_array = torch.stack([x[1] for x in reward_ratio_list]).cpu().numpy()  
                repeat_counts = (ratio_array * 10000000).astype(int)
                weighted_samples = np.repeat(advr_array, repeat_counts) 
                
                end = time.time()
                
                weighted_samples = np.array(weighted_samples)
                threshold0 = np.percentile(weighted_samples, self._cfgs.algo_cfgs.reward_buffer_percentile) 
                end = time.time()
                
            else:
                threshold0 = max(np.mean(adv_r1), np.percentile(adv_r1, self._cfgs.algo_cfgs.reward_buffer_percentile))
                end = time.time()
            
            exceedances = weighted_samples[weighted_samples > threshold0] - threshold0  
            if len(exceedances) > 0:  
                shape, loc, scale = genpareto.fit(exceedances)
                percentile = self.distribution_reward_percentile
                threshold0 = genpareto.ppf(percentile, shape, loc, scale) + threshold0  
            else:
                threshold0 = np.percentile(adv_r1, self._cfgs.algo_cfgs.reward_buffer_percentile)
            end = time.time()
            
        high_adv_indices = adv_r >= threshold0 

        
        for _ in range(self.oversample_factor):
            obs = torch.cat([obs, data['obs'][high_adv_indices]], axis=0)  
            act = torch.cat([act, data['act'][high_adv_indices]], axis=0)
            logp = torch.cat([logp, data['logp'][high_adv_indices]], axis=0)
            adv_r = torch.cat([adv_r, data['adv_r'][high_adv_indices]], axis=0)
            adv_c = torch.cat([adv_c, data['adv_c'][high_adv_indices]], axis=0)
            target_value_r = torch.cat([target_value_r, data['target_value_r'][high_adv_indices]], axis=0)
            target_value_c = torch.cat([target_value_c, data['target_value_c'][high_adv_indices]], axis=0)
        
       
        if self._cfgs.algo_cfgs.use_cost:
            
            samples = np.array(self._env.rollcost)
            
            
            obs_act_logp_cost = self._env.cost_obs_act_logp  
            
            if self.use_cost_buffer:
                
                for i in range(len(obs_act_logp_cost['cost'])):  
                    if len(self.cost_buffer_cost) >= self.buffer_size:
                        self.cost_buffer_cost.pop(0)
                        self.cost_buffer_obs.pop(0)
                        self.cost_buffer_act.pop(0)
                        self.cost_buffer_logp.pop(0)
                    self.cost_buffer_cost.append(obs_act_logp_cost['cost'][i]) 
                    self.cost_buffer_obs.append(rollobs[i])
                    self.cost_buffer_act.append(rollact[i])
                    self.cost_buffer_logp.append(rolllogp[i])
                   
            if len(self.cost_buffer_cost) > 0:
                   
                cost_buffer_obs = [torch.stack(obs,dim=0) for obs in self.cost_buffer_obs]  
                cost_buffer_act = [torch.stack(act,dim=0) for act in self.cost_buffer_act] 
               
                cost_buffer_logp = [torch.stack(logp,dim=0) for logp in self.cost_buffer_logp]
                
                logp_current = []
                
                
                with torch.no_grad():
                    for i in range(len(cost_buffer_act)):
                        tmp_dist = self._actor_critic.actor(cost_buffer_obs[i])  
                        logp_current.append(tmp_dist.log_prob(cost_buffer_act[i]).sum(axis=-1)) 
                    logp_current = torch.stack(logp_current,dim=0)
                   
                cost_buffer_logp = torch.stack(cost_buffer_logp,dim=0)
                
                logp_diff = torch.clamp((logp_current - cost_buffer_logp).sum(dim=1), min=0.8, max=1.2)
                
                ratio = torch.exp(logp_diff)/torch.exp(logp_diff).sum()
                
                cost_ratio_list = [(self.cost_buffer_cost[i], ratio[i]) for i in range(len(self.cost_buffer_cost))]
            
            
            if (np.all(samples == 0)):
                threshold = 0
            else:
                
                if len(cost_ratio_list) > 0:
                    
                    weighted_samples = []
                    for cost, ratio in cost_ratio_list:
                        weighted_samples.extend([cost] * int(ratio * 10000)) 
                    weighted_samples = np.array(weighted_samples)  
                   
                    threshold = np.percentile(weighted_samples, self._cfgs.algo_cfgs.cost_buffer_percentile)
                else:
                    threshold = np.percentile(samples, self._cfgs.algo_cfgs.cost_buffer_percentile)
               
                exceedances = weighted_samples[weighted_samples > threshold] - threshold  
                if len(exceedances) > 0:
                    shape, loc, scale = genpareto.fit(exceedances)
                    
                    threshold = genpareto.ppf(self.distribution_percentile, shape, loc, scale) + threshold
                else:
                    threshold = np.percentile(weighted_samples, self._cfgs.algo_cfgs.cost_buffer_percentile)
                
                common = np.percentile(weighted_samples, self._cfgs.algo_cfgs.cost_buffer_percentile)
                
                threshold = threshold + common
            self._logger.store(
                {
                    'Misc/threshold': threshold,
                },
            )
           
            self.ep_costs = threshold - self._cfgs.algo_cfgs.cost_limit
        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
            },
        )



    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:

        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        self._actor_critic.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grads = get_flat_gradients_from(self._actor_critic.actor)

       
        ep_costs = self.ep_costs  

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
          
            },
        )
