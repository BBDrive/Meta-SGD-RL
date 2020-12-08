import torch

from meta_sgd_rl.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)


class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 tau=1.0, clip_param=0.2, lr_ppo=5e-5, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.lr_ppo = lr_ppo
        self.to(device)

        self.model_params = list(self.policy.parameters()) + list(self.policy.meta_sgd_lr.values())
        self.optimizer = torch.optim.Adam(self.model_params, lr=self.lr_ppo)

    def inner_loss(self, episodes, params=None):  # return shape (1, batch_size)
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        episedes： 单个任务的所有trajectories
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
                              weights=episodes.mask)

        return loss

    def adapt(self, episodes):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        # 基于当前的parameters再更新
        params = self.policy.update_params(loss, self.device)

        return params

    def sample(self, tasks):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                                                 gamma=self.gamma, device=self.device)

            params = self.adapt(train_episodes)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                                                 gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def surrogate_loss(self, episodes, old_pis=None):
        losses, pis = [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            # 返回用reinforce更新一次的参数；例如第一次时actor参数没更新时返回的是sample valid_episodes的参数
            params = self.adapt(train_episodes)
            # with torch.set_grad_enabled(old_pi is None):
            pi = self.policy(valid_episodes.observations, params=params)
            pis.append(detach_distribution(pi))

            if old_pi is None:
                old_pi = detach_distribution(pi)

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages,
                                            weights=valid_episodes.mask)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advantages
            loss = torch.min(surr1, surr2)

            loss = -weighted_mean(loss, dim=0,
                                  weights=valid_episodes.mask)
            losses.append(loss)

        return (torch.mean(torch.stack(losses, dim=0)),
                pis)

    def step(self, episodes, ppo_update_time=5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        loss, old_pis = self.surrogate_loss(episodes)
        for i in range(ppo_update_time):

            # grads = torch.autograd.grad(loss, self.policy.parameters())
            # grads = parameters_to_vector(grads)
            # params = parameters_to_vector(self.policy.parameters())
            # vector_to_parameters(params - self.lr_ppo * grads,
            #                      self.policy.parameters())

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model_params, 0.5)
            self.optimizer.step()
            if i == (ppo_update_time - 1):
                break
            loss, _ = self.surrogate_loss(episodes, old_pis=old_pis)

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
