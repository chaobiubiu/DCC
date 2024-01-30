import copy
from src.components.episode_buffer import EpisodeBatch
from src.components.standarize_stream import RunningMeanStd
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
from src.modules.infer_networks.MINE import New_MINE
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop


# Binary classifier that classifies which skills the current (\tau^i, a^i) comes from.
class BinaryClassifier(nn.Module):
    def __init__(self, args):
        super(BinaryClassifier, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear((args.rnn_hidden_dim + args.n_actions), args.rnn_hidden_dim),
                                  nn.ReLU())
        self.out = nn.Linear(args.rnn_hidden_dim, args.n_skills)

    def forward(self, concat_batches):
        x1 = self.fc_1(concat_batches)
        pred_label = self.out(x1)
        return pred_label


def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


class DCCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        print("Use dcc_learner.")

        self.params = list(mac.parameters())        # high_agent + agent

        self.last_target_update_episode = 0

        # In this implementation, we use mixer for both high-level policy and low-level policy.
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

        self.primitive_mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.primitive_mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.primitive_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.primitive_mixer.parameters())
            self.target_primitive_mixer = copy.deepcopy(self.primitive_mixer)

        self.n_agents = args.n_agents
        self.n_skills = args.n_skills
        self.skill_interval = args.skill_interval

        self.Classifier = BinaryClassifier(args).to(self.args.device)
        self.params += list(self.Classifier.parameters())

        # maximize I(s,z^joint; a^joint)
        self.Mine_Policy = New_MINE(args.state_shape, args.n_skills, args.n_actions, self.n_agents).to(self.args.device)
        self.params += list(self.Mine_Policy.parameters())

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
        hidden_states_array = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, high_agent_outs, hidden_states = self.mac.train_forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_states_array.append(hidden_states)
            if t % self.skill_interval == 0 and t < batch.max_seq_length - 1:
                high_mac_out.append(high_agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time, shape=(bs, max_seq_length, n_agents, n_actions)
        high_mac_out = th.stack(high_mac_out, dim=1)  # Concat over time, shape=(bs, skills_at, n_agents, n_skills)
        hidden_states_array = th.stack(hidden_states_array[:-1], dim=1)     # shape=(bs, max_seq_length-1, n_agents, rnn_hidden_dim)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim, shape=(bs, max_seq_length-1, n_agents)
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
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

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
            # Low-level policy, log q(z^i | \tau^i, a^i)
            tau_detach = hidden_states_array.detach()  # (bs, max_seq_length-1, n_agents, rnn_hidden_dim)
            softmax_actions = th.softmax(mac_out[:, :-1], dim=-1).detach()  # (bs, max_seq_length-1, n_agents, n_actions)
            concat_tau_act = th.cat([tau_detach, softmax_actions], dim=-1).reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
            pred_labels = self.Classifier(concat_tau_act)  # (bs*(max_seq_length-1)*n_agents, rnn_hidden_dim+n_actions)
            # =========  More accurate prediction，softmax() and log_softmax() are bigger.
            log_pred_labels = th.log_softmax(pred_labels, dim=-1).reshape(batch.batch_size, (batch.max_seq_length - 1), self.n_agents, -1)
            target_labels = batch["skills"][:, :-1]  # (bs, (max_seq_length-1), n_agents, 1)
            intrinsic_rew_lower = th.gather(log_pred_labels, dim=-1, index=target_labels)       # (bs, max_seq_length-1, n_agents, 1)
            intrinsic_rew_lower = th.mean(intrinsic_rew_lower, dim=2) * self.args.low_per_weight  # (bs, max_seq_length-1, 1)

            # High_td_error as intrinsic rewards
            # high_signals = - high_td_error          # (bs, skills_at, 1)
            # high_signals = high_signals.unsqueeze(dim=2).expand(-1, -1, self.skill_interval, -1) / self.skill_interval
            # high_signals = high_signals.reshape(batch.batch_size, skills_t, -1)[:, :(batch.max_seq_length - 1)]
            # high_signals = high_signals * self.args.low_global_weight

            # ======= I(s, z^joint; a^joint)
            # joint_softmax_actions.shape=(bs*(max_seq_length-1), n_agents*n_actions)
            joint_softmax_actions = softmax_actions.reshape(-1, self.n_agents * self.args.n_actions)
            selected_skills = batch["skills"][:, :-1].squeeze(dim=-1)      # (bs, max_seq_length-1, n_agents)
            # shape=(bs*(max_seq_length-1), n_agents*n_skills)
            joint_onehot_skills = F.one_hot(selected_skills, num_classes=self.n_skills).reshape(-1, self.n_agents * self.n_skills).float()
            batch_states = batch["state"][:, :-1].reshape(-1, self.args.state_shape)
            neg_MI, half_MI = self.Mine_Policy(batch_states, joint_onehot_skills, joint_softmax_actions)
            mi_rewards = - neg_MI.detach()      # (bs*(max_seq_length-1), 1)
            mi_rewards = mi_rewards.reshape(batch.batch_size, (batch.max_seq_length - 1), 1) * self.args.low_global_weight

            # if self.args.anneal:
            #     if t_env > 1000000:
            #         intrinsic_rew_lower = max(1 - self.args.anneal_rate * (t_env - 1000000) / 1000000, 0) * intrinsic_rew_lower
            #         mi_rewards = max(1 - self.args.anneal_rate * (t_env - 1000000) / 1000000, 0) * mi_rewards

            # log_prob = - th.log_softmax(mac_out[:, :-1], dim=-1)      # (bs, max_seq_length-1, n_agents, n_actions)
            # selected_act_prob = th.gather(log_prob, dim=3, index=actions)     # (bs, max_seq_length-1, n_agents, 1)
            # entropy_rewards = th.mean(selected_act_prob, dim=2)     # (bs, (max_seq_length-1), 1)

            # High-level policy, H(p(z^i | \tau^i))
            # log_softmax_high_actions = th.log_softmax(high_mac_out, dim=-1)     # (bs, skills_at, n_agents, n_skills)
            # selected_log_softmax_actions = th.gather(log_softmax_high_actions, dim=-1, index=skills.long())
            # intrinsic_rew_high = th.mean(selected_log_softmax_actions, dim=2) * self.args.high_entropy_weight       # (bs, skills_at, 1)

        # ================= Record D_kl [p(a^i | \tau^i, z^i) || p(a^i | \tau^i)]
        # with th.no_grad():
        #     tau_rep = hidden_states_array.unsqueeze(dim=3).expand(-1, -1, -1, self.n_skills, -1)
        #     pseudo_skills = th.eye(self.n_skills, device=self.args.device).unsqueeze(dim=0)
        #     pseudo_skills = pseudo_skills.expand((batch.batch_size * (batch.max_seq_length - 1) * self.n_agents), -1, -1)
        #     pseudo_skills = pseudo_skills.reshape(batch.batch_size, -1, self.n_agents, self.n_skills, self.n_skills)
        #     pred_qs = self.mac.agent.predict(tau_rep, pseudo_skills)
        #     policy_pred_qs = th.softmax(pred_qs, dim=-1)        # (bs, max_seq_length-1, n_agents, n_skills, n_actions)
        #     mean_policy_pred_qs = th.mean(policy_pred_qs, dim=3)        # (bs, max_seq_length-1, n_agents, n_actions)
        #     kl_distance = (softmax_actions * th.log(softmax_actions / mean_policy_pred_qs)).sum(dim=-1, keepdim=True)

        if self.primitive_mixer is not None:
            chosen_action_qvals = self.primitive_mixer(chosen_action_qvals, batch["state"][:, :-1].detach())  # (bs, max_seq_length-1, 1)
            target_max_qvals = self.target_primitive_mixer(target_max_qvals, batch["state"][:, 1:].detach())

        # Calculate one-step Q-Learning targets for low-level CTDE learner.
        if self.args.use_intrinsic:
            shaped_rewards = rewards + intrinsic_rew_lower + mi_rewards
        else:
            shaped_rewards = rewards

        targets = shaped_rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)
        mask_shape = list(mask.shape)
        mask_shape[1] = skills_t
        high_mask = th.zeros(mask_shape, device=self.args.device)
        high_mask[:, :mask.shape[1]] = mask.detach().clone()
        high_mask = high_mask.view(batch.batch_size, skills_at, self.skill_interval, -1)[:, :, 0]  # mask仍然是选取decision interval处的mask info

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        masked_high_td_error = high_td_error * high_mask

        if self.args.use_discrim:
            # ======================= Classify Loss =======================
            pred_labels = self.Classifier(concat_tau_act)  # (bs*(max_seq_length-1)*n_agents, rnn_hidden_dim+n_actions)
            target_labels = target_labels.reshape(-1).long()
            discrim_loss = F.cross_entropy(pred_labels, target_labels, reduction='none')
            discrim_loss = discrim_loss.reshape(batch.batch_size, (batch.max_seq_length - 1), self.n_agents, -1)
            discrim_loss = th.mean(discrim_loss, dim=2)     # (bs, max_seq_length-1, 1)
            discrim_loss = (discrim_loss * mask).sum() / mask.sum()

            # Maximize I(s, z^joint; a^joint) when high-td-error > 0
            high_signals_detach = - high_td_error.detach()  # (bs, skills_at, 1)
            high_signals_detach = high_signals_detach.unsqueeze(dim=2).expand(-1, -1, self.skill_interval, -1)
            high_signals_detach = high_signals_detach.reshape(-1, skills_t, 1)[:, :(batch.max_seq_length - 1)]
            positive_labels = (high_signals_detach > 0.0).float()
            pos_loaa, pos_MI = self.Mine_Policy(batch_states, joint_onehot_skills, joint_softmax_actions)
            pos_loaa = pos_loaa.reshape(batch.batch_size, (batch.max_seq_length - 1), 1)
            mi_loss = (pos_loaa * mask * positive_labels).sum() / ((mask * positive_labels).sum() + th.tensor(1e-8, device=self.args.device))
        else:
            discrim_loss = None
            mi_loss = None

        # Normal L2 loss, take mean over actual data
        low_td_loss = (masked_td_error ** 2).sum() / mask.sum()
        high_loss = (masked_high_td_error ** 2).sum() / high_mask.sum()

        if discrim_loss is not None:
            discrim_loss = discrim_loss * self.args.discrim_loss_weight
            mi_loss = mi_loss * self.args.discrim_loss_weight
            loss = low_td_loss + high_loss + discrim_loss + mi_loss
        else:
            loss = low_td_loss + high_loss

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
            if discrim_loss is not None:
                self.logger.log_stat("discrim_loss", discrim_loss.item(), t_env)
            if mi_loss is not None:
                self.logger.log_stat("mi_loss", mi_loss.item(), t_env)

            # self.logger.log_stat("kl_distance_mean", kl_distance.mean(), t_env)

            self.logger.log_stat("intrinsic_rew_lower_mean", intrinsic_rew_lower.mean().item(), t_env)
            # self.logger.log_stat("high_signals_mean", high_signals.mean(), t_env)
            self.logger.log_stat("mi_rewards_mean", mi_rewards.mean().item(), t_env)
            # self.logger.log_stat("entropy_rewards_mean", entropy_rewards.mean(), t_env)
            self.logger.log_stat("env_reward_mean", rewards.mean().item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)

            high_mask_elems = high_mask.sum().item()
            self.logger.log_stat("high_td_error_abs", (masked_high_td_error.abs().sum().item()/high_mask_elems), t_env)
            self.logger.log_stat("high_q_taken_mean", (chosen_skill_qvals * high_mask).sum().item()/(high_mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("high_target_mean", (high_targets * high_mask).sum().item()/(high_mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.primitive_mixer is not None:
            self.target_primitive_mixer.load_state_dict(self.primitive_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")


    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)
        if self.primitive_mixer is not None:
            self.primitive_mixer.to(self.args.device)
            self.target_primitive_mixer.to(self.args.device)


    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.primitive_mixer is not None:
            th.save(self.primitive_mixer.state_dict(), "{}/primitive_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        if self.primitive_mixer is not None:
            self.primitive_mixer.load_state_dict(th.load("{}/primitive_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))