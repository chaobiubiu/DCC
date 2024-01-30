import numpy as np


class SoftDecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 pretrain_time,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        self.pretrain_time = pretrain_time
        self.reset = True
        self.start_t = 0

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        # If employ forward() in learner.py, T is set None
        if T is None:
            return 0.05

        # When current time step >= pretrain_time, explore with epsilon greedy policy.
        # Before it, the high-level policy outputs skills for low-level policy uniformly.
        if T > self.pretrain_time and self.reset:
            self.reset = False
            self.time_length = self.time_length
            self.delta = (self.start - self.finish) / self.time_length
            self.start_t = T

        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * (T - self.start_t))
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

