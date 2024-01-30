import copy
from src.components.episode_buffer import EpisodeBatch
from src.components.standarize_stream import RunningMeanStd
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop


# LSTM Examples::
#         >>> rnn = nn.LSTM(10, 20, 2)
#         >>> input = torch.randn(5, 3, 10)
#         >>> h0 = torch.randn(2, 3, 20)
#         >>> c0 = torch.randn(2, 3, 20)
#         >>> output, (hn, cn) = rnn(input, (h0, c0))


# A classifier that identifies which skill the current tau segment comes from.
class Decoder(nn.Module):
    def __init__(self, input_shape, args):
        super(Decoder, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.fc_1 = nn.Sequential(nn.Linear(input_shape, int(args.rnn_hidden_dim // 2)), nn.ReLU())
        # Bi-directional LSTM
        # **input** of shape `(seq_len, batch, input_size)`: tensor containing the features of the input sequence.
        self.lstm = nn.LSTM(int(args.rnn_hidden_dim // 2), int(args.rnn_hidden_dim // 2), 1, bidirectional=True)
        # Owing to the bi-directional LSTM, the input of the layer below becomes (args.rnn_hidden_dim // 2) * 2
        self.out = nn.Linear(args.rnn_hidden_dim, args.n_skills)

    def forward(self, obs_sequences):
        # Input.shape may be (bs, skills_at, skill_interval, n_agents, input_shape) ==> (skill_interval, bs*skills_at*n_agents, input_shape)
        obs_sequences = obs_sequences.permute(2, 0, 1, 3, 4).reshape(self.args.skill_interval, -1, self.input_shape)
        obs_sequences = obs_sequences[1:, :, :] - obs_sequences[:-1, :, :]      # (skills_interval-1, bs*skills_at*n_agents, input_shape)?
        # Intialize h0 and c0, the first vector represents: num_layers*num_directions (2 if bidirectional else 1)
        h0 = th.zeros((2, obs_sequences.size(1), int(self.args.rnn_hidden_dim // 2)), device=self.args.device)
        c0 = th.zeros((2, obs_sequences.size(1), int(self.args.rnn_hidden_dim // 2)), device=self.args.device)
        # TODO: check the shape of inputs
        x1 = self.fc_1(obs_sequences)
        x2, _ = self.lstm(x1, (h0, c0))     # x2.shape=(seq_length, bs*skills_at*n_agents, hidden_size * 2)
        # x2.shape=(skills_interval-1, bs*skills_at*n_agents, hidden_size*2)
        # **output** of shape `(seq_len, batch, num_directions * hidden_size)`
        # The outputs are mean-pooled over time and passes through a softmax output layer to produce probabilities over skills.
        average_x2 = th.mean(x2, dim=0, keepdim=True).transpose(1, 0).squeeze(dim=1)       # (bs*skills_at*n_agents, hidden_size * 2)
        pred_label = self.out(average_x2)       # (bs*skills_at*n_agents, n_skills)
        return pred_label


class HSDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        print("Use hsd_learner.")

        self.params = list(mac.parameters())        # high_agent + agent

        self.last_target_update_episode = 0

        # In HSD, we only use mixer for the high-level policy.
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # The low-level policy uses IQL.

        self.n_agents = args.n_agents
        self.n_skills = args.n_skills
        self.skill_interval = args.skill_interval

        self.Decoder = Decoder(mac.input_shape, args).to(self.args.device)
        self.params += list(self.Decoder.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        if self.args.standardise_rewards:
            print("Use standardise_reward settings.")
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.args.device)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # onehot_actions = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # high-level policy
        skills_shape_o = batch["skills"][:, :-1].shape      # (bs, max_seq_length-1, n_agents, 1)
        skills_at = int(np.ceil(skills_shape_o[1] / self.skill_interval))     # (max_seq_length-1 / skill_interval)
        skills_t = skills_at * self.skill_interval

        skills_shape = list(skills_shape_o)
        skills_shape[1] = skills_t      # (bs, skills_t, n_agents, 1)
        skills = th.zeros(skills_shape, dtype=batch["skills"].dtype, device=self.args.device)
        skills[:, :skills_shape_o[1]] = batch["skills"][:, :-1]
        skills_all = skills.view(batch.batch_size, skills_at, self.skill_interval, self.n_agents, -1)  # shape=(bs, skills_at, skill_interval, n_agents, 1)
        skills = skills_all[:, :, 0]  # shape=(bs, skills_at, n_agents, 1)

        # Calculate estimated Q-Values
        mac_out = []
        high_mac_out = []
        obs_array = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, high_agent_outs, local_obs = self.mac.train_forward(batch, t=t)
            mac_out.append(agent_outs)
            obs_array.append(local_obs)
            if t % self.skill_interval == 0 and t < batch.max_seq_length - 1:
                high_mac_out.append(high_agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time, shape=(bs, max_seq_length, n_agents, n_actions)
        high_mac_out = th.stack(high_mac_out, dim=1)  # Concat over time, shape=(bs, skills_at, n_agents, n_skills)
        obs_array = th.stack(obs_array[:-1], dim=1)     # shape=(bs, max_seq_length-1, n_agents, input_shape)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions)  # Remove the last dim, shape=(bs, max_seq_length-1, n_agents, 1)
        chosen_skill_qvals = th.gather(high_mac_out, dim=3, index=skills.long()).squeeze(3)  # shape=(bs, skills_at, n_agents)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_high_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_high_agent_outs, _ = self.target_mac.train_forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            if t % self.skill_interval == 0 and t < batch.max_seq_length - 1:
                target_high_mac_out.append(target_high_agent_outs)

        target_high_mac_out.append(th.zeros((batch.batch_size, self.n_agents, self.n_skills), device=self.args.device))    # skills_at+1 个 (bs, n_agents, n_skills)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, shape=(bs, max_seq_length-1, n_agents, n_actions)
        target_high_mac_out = th.stack(target_high_mac_out[1:], dim=1)    # Concat over time, shape=(bs, skills_at, n_agents, n_skills)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions)     # (bs, max_seq_length-1, n_agents, 1)

            high_mac_out_detach = high_mac_out.clone().detach()
            high_mac_out_detach = th.cat([high_mac_out_detach[:, 1:], high_mac_out_detach[:, 0: 1]], dim=1)    # shape=(bs, skills_at, n_agents, n_skills)
            cur_max_skills = high_mac_out_detach.max(dim=3, keepdim=True)[1]
            target_skill_max_qvals = th.gather(target_high_mac_out, 3, cur_max_skills).squeeze(3)      # shape=(bs, skills_at, n_agents)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_skill_max_qvals = target_high_mac_out.max(dim=3)[0]

        # Mix
        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        #     target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        state_shape_o = batch["state"][:, :-1].shape
        state_shape = list(state_shape_o)
        state_shape[1] = skills_t
        high_states = th.zeros(state_shape, device=self.args.device)
        high_states[:, :state_shape_o[1]] = batch["state"][:, :-1].detach().clone()
        high_states = high_states.view(batch.batch_size, skills_at, self.skill_interval, -1)[:, :, 0]  # shape=(bs, skills_at, state_shape)

        if self.mixer is not None:
            chosen_skill_qvals = self.mixer(chosen_skill_qvals, high_states)
            next_high_states = th.cat([high_states[:, 1:], high_states[:, 0:1]], dim=1)     # A rude operation to shape next high_states.
            target_skill_max_qvals = self.target_mixer(target_skill_max_qvals, next_high_states)

        # Calculate one-step Q-Learning targets for high-level subtask policy (CTDE learner)
        rewards_shape = list(rewards.shape)
        rewards_shape[1] = skills_t
        high_rewards = th.zeros(rewards_shape, device=self.args.device)
        high_rewards[:, :rewards.shape[1]] = rewards.detach().clone()
        high_rewards = high_rewards.view(batch.batch_size, skills_at, self.skill_interval).sum(dim=-1, keepdim=True)  # (bs, skills_at, 1)

        # role_terminated
        terminated_shape_o = terminated.shape
        terminated_shape = list(terminated_shape_o)
        terminated_shape[1] = skills_t
        high_terminated = th.zeros(terminated_shape, device=self.args.device)
        high_terminated[:, :terminated_shape_o[1]] = terminated.detach().clone()
        high_terminated = high_terminated.view(batch.batch_size, skills_at, self.skill_interval).sum(dim=-1, keepdim=True)  # 注意这里，并非取role_interval决策节点处的terminated Info，而是考虑每个role_interval期间是否有terminated label role_terminated

        high_targets = high_rewards + self.args.gamma * (th.tensor(1.0, device=self.args.device) - high_terminated) * target_skill_max_qvals
        high_td_error = (chosen_skill_qvals - high_targets.detach())        # (bs, skills_at, 1)

        # ================= Calculate intrinsic rewards for low-level policy ==================
        with th.no_grad():
            # Low-level policy, log q(z^i | o_{t}^{i}, o_{t+1}^{i}, ..., o_{t+k}^{i})
            # obs_array.shape=(bs, max_seq_length-1, n_agents, input_shape)
            obs_shape_o = list(obs_array.shape)
            obs_shape_o[1] = skills_t
            obs_sequence = th.zeros(obs_shape_o, device=self.args.device)       # (bs, skills_t, n_agents, input_shape)
            obs_sequence[:, :obs_array.shape[1]] = obs_array.detach().clone()
            obs_sequence = obs_sequence.view(batch.batch_size, skills_at, self.skill_interval, self.n_agents, -1)
            # Then we use the classifier to output the probabilities.
            pred_labels = self.Decoder(obs_sequence)        # (bs*skills_at*n_agents, n_skills)?
            pred_probs = th.softmax(pred_labels, dim=-1)        # shape=(bs*skills_at*n_agents, n_skills)
            pred_probs = pred_probs.view(batch.batch_size, skills_at, self.n_agents, self.n_skills)

            intrinsic_rew = th.gather(pred_probs, dim=-1, index=skills.long())       # (bs, skills_at, n_agents, 1)

            # Calculate one-step Q-Learning targets for low-level CTDE learner.
            # rewards.shape=(bs, max_seq_length-1, 1)
            rep_rewards = rewards.unsqueeze(dim=2).expand(-1, -1, self.n_agents, -1)        # (bs, max_seq_length-1, n_agents, 1)
            # intrinsic_rew.shape=(bs, skills_at, n_agents, 1)
            intrinsic_rew = intrinsic_rew.unsqueeze(dim=2)      # (bs, skills_at, 1, n_agents, 1)
            padding_intrinsic = th.zeros_like(intrinsic_rew, device=self.args.device).expand(-1, -1, (self.skill_interval - 1), -1, -1)
            # We only use the intrinsic reward at the last of each segment.
            intrinsic_rew = th.cat([padding_intrinsic, intrinsic_rew], dim=2).view(batch.batch_size, skills_t, self.n_agents, 1)[:, :(batch.max_seq_length - 1)]

        shaped_rewards = self.args.intrinsic_alpha * rep_rewards + (1 - self.args.intrinsic_alpha) * intrinsic_rew      # (bs, max_seq_length-1, n_agents, 1)

        # The low-level policy employs iql.
        # chosen_action_qvals.shape=(bs, max_seq_length-1, n_agents, 1), target_max_qvals.shape=(bs, max_seq_length-1, n_agents, 1)
        rep_terminated = terminated.unsqueeze(dim=2).expand(-1, -1, self.n_agents, -1)
        targets = shaped_rewards + self.args.gamma * (1 - rep_terminated) * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach())     # (bs, max_seq_length-1, n_agents, 1)

        # mask = mask.expand_as(td_error), mask.shape=(bs, max_seq_length-1, 1)
        rep_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.n_agents, -1)      # (bs, max_seq_length-1, n_agents, 1)
        mask_shape = list(mask.shape)
        mask_shape[1] = skills_t
        high_mask = th.zeros(mask_shape, device=self.args.device)
        high_mask[:, :mask.shape[1]] = mask.detach().clone()
        high_mask = high_mask.view(batch.batch_size, skills_at, self.skill_interval, -1)[:, :, 0]  # mask仍然是选取decision interval处的mask info

        # 0-out the targets that came from padded data
        masked_td_error = td_error * rep_mask
        masked_high_td_error = high_td_error * high_mask

        # Normal L2 loss, take mean over actual data
        low_td_loss = (masked_td_error ** 2).sum() / rep_mask.sum()
        high_loss = (masked_high_td_error ** 2).sum() / high_mask.sum()

        # ======================= Classify Loss =======================
        pred_labels = self.Decoder(obs_sequence)        # shape=(bs*skills_at*n_agents, n_skills)?
        target_labels = skills.reshape(-1).long()       # shape=(bs*skills_at*n_agents)
        discrim_loss = F.cross_entropy(pred_labels, target_labels)

        loss = low_td_loss + high_loss + discrim_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("low_td_loss", low_td_loss.item(), t_env)
            self.logger.log_stat("high_loss", high_loss.item(), t_env)
            self.logger.log_stat("discrim_loss", discrim_loss.item(), t_env)

            self.logger.log_stat("intrinsic_rew_mean", intrinsic_rew.mean().item(), t_env)
            self.logger.log_stat("env_reward_mean", rewards.mean().item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)

            mask_elems = rep_mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * rep_mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * rep_mask).sum().item() / (mask_elems * self.args.n_agents), t_env)

            high_mask_elems = high_mask.sum().item()
            self.logger.log_stat("high_td_error_abs", (masked_high_td_error.abs().sum().item()/high_mask_elems), t_env)
            self.logger.log_stat("high_q_taken_mean", (chosen_skill_qvals * high_mask).sum().item()/(high_mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("high_target_mean", (high_targets * high_mask).sum().item()/(high_mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")


    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)


    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.Decoder.state_dict(), "{}/decoder.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))