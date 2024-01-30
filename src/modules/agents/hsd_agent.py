import torch as th
import torch.nn as nn
import torch.nn.functional as F


class HSDAgent(nn.Module):
    def __init__(self, input_shape, n_skills, args):
        super(HSDAgent, self).__init__()
        self.args = args
        print("Use hsd agent network.")

        self.fc1 = nn.Linear((input_shape + n_skills), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_out = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, assigned_skills):
        inps = th.cat([inputs, assigned_skills], dim=-1)
        x = F.relu(self.fc1(inps))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc_out(h)

        return q, h