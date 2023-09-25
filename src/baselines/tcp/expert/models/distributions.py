from typing import Optional, Tuple
import torch as th
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.nn import functional as F
import numpy as np


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class BetaDistribution():
    def __init__(self, action_dim=2, dist_init=None):
        assert action_dim == 2

        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0

        # [beta, alpha], [0, 1]
        self.acc_exploration_dist = {
            # [1, 2.5]
            # [1.5, 1.0]
            'go': th.FloatTensor([1.0, 2.5]),
            'stop': th.FloatTensor([1.5, 1.0])
        }
        self.steer_exploration_dist = {
            'turn': th.FloatTensor([1.0, 1.0]),
            'straight': th.FloatTensor([3.0, 3.0])
        }

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:

        linear_alpha = nn.Linear(latent_dim, self.action_dim)
        linear_beta = nn.Linear(latent_dim, self.action_dim)

        if self.dist_init is not None:
            # linear_alpha.weight.data.fill_(0.01)
            # linear_beta.weight.data.fill_(0.01)
            # acc
            linear_alpha.bias.data[0] = self.dist_init[0][1]
            linear_beta.bias.data[0] = self.dist_init[0][0]
            # steer
            linear_alpha.bias.data[1] = self.dist_init[1][1]
            linear_beta.bias.data[1] = self.dist_init[1][0]

        alpha = nn.Sequential(linear_alpha, nn.Softplus())
        beta = nn.Sequential(linear_beta, nn.Softplus())
        return alpha, beta

    def proba_distribution(self, alpha, beta):
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> th.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return th.mean(entropy_loss)

    def exploration_loss(self, exploration_suggests) -> th.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        alpha = self.distribution.concentration1.detach().clone()
        beta = self.distribution.concentration0.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                beta[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                alpha[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                beta[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                alpha[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Beta(alpha, beta)

        exploration_loss = th.distributions.kl_divergence(self.distribution, dist_ent)
        return th.mean(exploration_loss)

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        x = th.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = self.distribution.mean[mask4]

        return x

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()
