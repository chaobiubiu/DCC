from src.modules.agents import REGISTRY as agent_REGISTRY
from src.modules.high_agents import REGISTRY as high_agent_REGISTRY
from src.components.action_selectors import REGISTRY as action_REGISTRY
import copy
import torch as th
import torch.nn.functional as F


# This multi-agent controller shares parameters between agents
class HSDMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents

        self.n_skills = args.n_skills
        self.skill_interval = args.skill_interval

        # High-level policy assigns skills, input=local_obs+last_action+agent_id, output=n_skills
        input_shape = self._get_input_shape(scheme)
        self.input_shape = input_shape

        self._build_high_agents(input_shape)

        # Low-level agent, input=local_obs+last_action+agent_id, assigned_skills are used to shape hyper-networks, output=n_actions
        self._build_low_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        # The high-level and low-level exploration policies.
        self.high_selector = action_REGISTRY[args.action_selector](args)
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.high_hidden_states = None
        self.hidden_states = None
        self.selected_skills = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, high_agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env=t_env)
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, t_env, test_mode=test_mode)
        return chosen_actions, self.selected_skills

    def forward(self, ep_batch, t, test_mode=False, t_env=None):
        original_inputs = self._build_inputs(ep_batch, t)      # shape=(bs*n_agents, -1)

        # Select skills with the high-level policy
        high_agent_outputs = None
        # The high-level policy selects skills per skill_interval time steps.
        if t % self.skill_interval == 0:
            avail_skills = th.ones((ep_batch.batch_size, self.n_agents, self.n_skills), device=self.args.device)        # (bs, n_agents, n_skills)
            high_agent_outputs, self.high_hidden_states = self.high_agent(original_inputs, self.high_hidden_states)     # (bs*n_agents, n_skills)
            high_agent_outputs = high_agent_outputs.reshape(ep_batch.batch_size, self.n_agents, -1)     # (bs, n_agents, n_skills)
            self.selected_skills = self.high_selector.select_action(high_agent_outputs, avail_skills, t_env, test_mode=test_mode)     # (bs, n_agents)

        # Compute utility value functions for low-level primitive actions.
        assigned_skills = F.one_hot(self.selected_skills.detach(), num_classes=self.n_skills).to(th.float32)        # (bs, n_agents, n_skills)
        assigned_skills = assigned_skills.reshape(-1, self.n_skills)     # (bs*n_agents, n_skills)
        agent_outputs, self.hidden_states = self.agent(original_inputs, self.hidden_states, assigned_skills.detach())       # (bs*n_agents, n_actions)

        return agent_outputs.view(ep_batch.batch_size, self.n_agents, -1), \
               (None if high_agent_outputs is None else high_agent_outputs)

    def train_forward(self, ep_batch, t):
        original_inputs = self._build_inputs(ep_batch, t)  # shape=(bs*n_agents, -1)

        # Select skills with the high-level policy
        high_agent_outputs = None
        # The high-level policy selects skills per skill_interval time steps.
        if t % self.skill_interval == 0:
            high_agent_outputs, self.high_hidden_states = self.high_agent(original_inputs, self.high_hidden_states)  # (bs*n_agents, n_skills)
            high_agent_outputs = high_agent_outputs.reshape(ep_batch.batch_size, self.n_agents, -1)  # (bs, n_agents, n_skills)
            self.selected_skills = ep_batch["skills"][:, t]     # (bs, n_agents, 1)

        # Compute utility value functions for low-level primitive actions.
        assigned_skills = F.one_hot(self.selected_skills.squeeze(dim=-1).detach(), num_classes=self.n_skills).to(th.float32)  # (bs, n_agents, n_skills)
        assigned_skills = assigned_skills.reshape(-1, self.n_skills)  # (bs*n_agents, n_skills)
        agent_outputs, self.hidden_states = self.agent(original_inputs, self.hidden_states, assigned_skills.detach())  # (bs*n_agents, n_actions)

        return agent_outputs.view(ep_batch.batch_size, self.n_agents, -1), \
               (None if high_agent_outputs is None else high_agent_outputs), \
               original_inputs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.high_hidden_states = self.high_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)    # bav
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return list(self.high_agent.parameters()) + list(self.agent.parameters())

    def load_state(self, other_mac):
        self.high_agent.load_state_dict(other_mac.high_agent.state_dict())
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.high_agent.to(self.args.device)
        self.agent.to(self.args.device)

    def save_models(self, path):
        th.save(self.high_agent.state_dict(), "{}/high_agent.th".format(path))
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.high_agent.load_state_dict(th.load("{}/high_agent.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_high_agents(self, input_shape):
        assert self.args.high_agent == "rnn"
        self.high_agent = high_agent_REGISTRY[self.args.high_agent](input_shape, self.n_skills, self.args)

    def _build_low_agents(self, input_shape):
        assert self.args.low_agent == "hsd_agent"
        self.agent = agent_REGISTRY[self.args.low_agent](input_shape, self.n_skills, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape