import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import SoftDecayThenFlatSchedule

REGISTRY = {}


class SoftEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.pretrain_time = int(args.t_max * args.pretrain_percentage)

        self.schedule = SoftDecayThenFlatSchedule(args.epsilon_start,
                                                  args.epsilon_finish,
                                                  args.epsilon_anneal_time,
                                                  self.pretrain_time,
                                                  decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        if t_env is not None and t_env < self.pretrain_time:
            # If pretrain the low-level skills, high-level policy randomly outputs skills for low-level policy.
            picked_actions = random_actions
        else:
            # Otherwise high-level policy outputs skills with epsilon-greedy exploration policy.
            picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        return picked_actions


REGISTRY["soft_epsilon_greedy"] = SoftEpsilonGreedyActionSelector