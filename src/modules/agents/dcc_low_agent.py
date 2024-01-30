import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DCCAgent(nn.Module):
    def __init__(self, input_shape, n_skills, args):
        super(DCCAgent, self).__init__()
        self.args = args
        self.is_latent_detach = args.is_latent_detach
        print("Use hyper-net-based utility function.")

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # Hyper-networks to generate weights and biases of the last layers.
        hidden_dim = int(args.rnn_hidden_dim // 2)
        # \tau^i, z^i --> w, b
        self.latent_net = nn.Sequential(nn.Linear((args.rnn_hidden_dim + n_skills), hidden_dim),
                                        nn.ReLU())
        self.fc2_w_net = nn.Linear(hidden_dim, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_net = nn.Linear(hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, assigned_skills):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        h_detach = h.detach()
        concat_inputs = th.cat([h_detach, assigned_skills], dim=-1)
        latent = self.latent_net(concat_inputs)       # (bs*n_agents, hidden_dim)

        if self.is_latent_detach:
            in_latent = latent.detach()
        else:
            in_latent = latent

        fc2_w = self.fc2_w_net(in_latent)      # (bs*n_agents, rnn_hidden_dim*n_actions)
        fc2_b = self.fc2_b_net(in_latent)      # (bs*n_agents, n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)

        shaped_h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(shaped_h, fc2_w) + fc2_b
        q = q.reshape(-1, self.args.n_actions)      # (bs*n_agents, n_actions)

        return q, h

    def predict(self, hidden_state, assigned_skills):
        bs, t, n_agents, n_skills, n_skills = assigned_skills.size()
        h_detach = hidden_state.detach()
        concat_inps = th.cat([h_detach, assigned_skills], dim=-1)
        latent = self.latent_net(concat_inps)  # (bs, t, n_agents, n_skills, hidden_dim)

        if self.is_latent_detach:
            in_latent = latent.detach()
        else:
            in_latent = latent

        fc2_w = self.fc2_w_net(in_latent)  # (bs, t, n_agents, n_skills, rnn_hidden_dim*n_actions)
        fc2_b = self.fc2_b_net(in_latent)  # (bs, t, n_agents, n_skills, n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)

        shaped_h = hidden_state.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(shaped_h, fc2_w) + fc2_b
        q = q.reshape(bs, t, n_agents, n_skills, self.args.n_actions)  # (bs*t*n_agents*n_skills, n_actions)

        return q


# class DCCAgent(nn.Module):
#     def __init__(self, input_shape, n_skills, args):
#         super(DCCAgent, self).__init__()
#         self.args = args
#         print("Use normal input utility network.")
#
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc_out = nn.Linear((args.rnn_hidden_dim + n_skills), args.n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
#
#     def forward(self, inputs, hidden_state, assigned_skills):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         concat_inps = th.cat([h, assigned_skills], dim=-1)
#         q = self.fc_out(concat_inps)
#
#         return q, h
#
#     def predict(self, hidden_state, assigned_skills):
#         concat_inps = th.cat([hidden_state, assigned_skills], dim=-1)
#         q = self.fc_out(concat_inps)
#         return q

# class DCCAgent(nn.Module):
#     def __init__(self, input_shape, n_skills, args):
#         super(DCCAgent, self).__init__()
#         self.args = args
#         self.is_latent_detach = args.is_latent_detach
#         print("Use skill-input original.")
#
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#
#         # Hyper-networks to generate weights and biases of the last layers.
#         hidden_dim = int(args.rnn_hidden_dim // 2)
#         # z^i --> w, b
#         self.latent_net = nn.Sequential(nn.Linear(n_skills, hidden_dim),
#                                         nn.ReLU())
#         self.fc2_w_net = nn.Linear(hidden_dim, args.rnn_hidden_dim * args.n_actions)
#         self.fc2_b_net = nn.Linear(hidden_dim, args.n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
#
#     def forward(self, inputs, hidden_state, assigned_skills):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#
#         latent = self.latent_net(assigned_skills)       # (bs*n_agents, hidden_dim)
#
#         if self.is_latent_detach:
#             in_latent = latent.detach()
#         else:
#             in_latent = latent
#
#         fc2_w = self.fc2_w_net(in_latent)      # (bs*n_agents, rnn_hidden_dim*n_actions)
#         fc2_b = self.fc2_b_net(in_latent)      # (bs*n_agents, n_actions)
#         fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
#         fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)
#
#         shaped_h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
#         q = th.bmm(shaped_h, fc2_w) + fc2_b
#         q = q.reshape(-1, self.args.n_actions)      # (bs*n_agents, n_actions)
#
#         return q, h
#
#     def predict(self, hidden_state, assigned_skills):
#         bs, t, n_agents, n_skills, n_skills = assigned_skills.size()
#         latent = self.latent_net(assigned_skills)  # (bs, t, n_agents, n_skills, hidden_dim)
#
#         if self.is_latent_detach:
#             in_latent = latent.detach()
#         else:
#             in_latent = latent
#
#         fc2_w = self.fc2_w_net(in_latent)  # (bs, t, n_agents, n_skills, rnn_hidden_dim*n_actions)
#         fc2_b = self.fc2_b_net(in_latent)  # (bs, t, n_agents, n_skills, n_actions)
#         fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
#         fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)
#
#         shaped_h = hidden_state.reshape(-1, 1, self.args.rnn_hidden_dim)
#         q = th.bmm(shaped_h, fc2_w) + fc2_b
#         q = q.reshape(bs, t, n_agents, n_skills, self.args.n_actions)  # (bs*t*n_agents*n_skills, n_actions)
#
#         return q


# class DCCAgent(nn.Module):
#     def __init__(self, input_shape, n_skills, args):
#         super(DCCAgent, self).__init__()
#         self.args = args
#         self.is_latent_detach = args.is_latent_detach
#         print("Use skill-input hypernet-based utility network.")
#
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#
#         # Hyper-networks to generate weights and biases of the last layers.
#         hidden_dim = int(args.rnn_hidden_dim // 2)
#         # z^i --> w, b
#         self.fc2_w_net = nn.Sequential(nn.Linear(n_skills, hidden_dim),
#                                         nn.ReLU(),
#                                         nn.Linear(hidden_dim, args.rnn_hidden_dim * args.n_actions))
#         self.fc2_b_net = nn.Sequential(nn.Linear(n_skills, hidden_dim),
#                                        nn.ReLU(),
#                                        nn.Linear(hidden_dim, args.n_actions))
#         # self.fc2_w_net = nn.Linear(hidden_dim, args.rnn_hidden_dim * args.n_actions)
#         # self.fc2_b_net = nn.Linear(hidden_dim, args.n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
#
#     def forward(self, inputs, hidden_state, assigned_skills):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#
#         fc2_w = self.fc2_w_net(assigned_skills)      # (bs*n_agents, rnn_hidden_dim*n_actions)
#         fc2_b = self.fc2_b_net(assigned_skills)      # (bs*n_agents, n_actions)
#         fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
#         fc2_b = fc2_b.reshape(-1, 1, self.args.n_actions)
#
#         shaped_h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
#         q = th.bmm(shaped_h, fc2_w) + fc2_b
#         q = q.reshape(-1, self.args.n_actions)      # (bs*n_agents, n_actions)
#
#         return q, h