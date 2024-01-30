import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        # Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -th.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - th.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1 == 2

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        # Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((th.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = th.clamp(q_samples, -1e6, 9.5)

        # print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = th.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = th.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1 == 2

    if average:
        return Eq.mean()
    else:
        return Eq


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
        Note that vectors should be sent as 1x1.
        Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
        Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
        Args:
            l: Local feature map.
            m: Multiple globals feature map.
            measure: f-divergence measure.
        Returns:
            torch.Tensor: Loss.
    '''
    N, units = l.size()

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = th.mm(m, l.t())     # shape=(N, N)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    # 对角线上的样本为正样本？其余位置为负样本？
    mask = th.eye(N).to(l.device)
    n_mask = th.tensor(1, device=l.device) - mask       # 对角线为0，其余位置不为0
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1)      # 仅包含对角线上的样本
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    loss = E_neg_term - E_pos_term
    return loss, MI


class New_MINE(nn.Module):
    def __init__(self, state_size, n_skills, n_actions, n_agents, measure="JSD"):
        super(New_MINE, self).__init__()
        self.measure = measure
        self.activation = F.leaky_relu
        self.fc_s = nn.Linear((state_size + n_skills * n_agents), 32)
        self.fc_a = nn.Linear((n_actions * n_agents), 32)

    def forward(self, state, joint_skills, joint_actions, params=None):
        # state.shape=(bs*t, state_shape)
        # joint_skills.shape=(bs*t, n_skills*n_agents), joint_actions.shape=(bs*t, n_actions*n_agents)
        state_skills = th.cat([state, joint_skills], dim=-1)        # (bs*t, state_shape+n_skills*n_agents)
        em_1 = self.activation(self.fc_s(state_skills), inplace=True)
        em_2 = self.activation(self.fc_a(joint_actions), inplace=True)
        agents_two_embeddings = [em_1, em_2]
        loss, MI = fenchel_dual_loss(agents_two_embeddings[0], agents_two_embeddings[1], measure=self.measure)
        return loss, MI