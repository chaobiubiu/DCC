from src.envs import REGISTRY as env_REGISTRY
from functools import partial
from src.components.episode_buffer import EpisodeBatch
import numpy as np
import time


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        assert self.batch_size == 1

        if 'stag_hunt' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        elif 'mpe' in self.args.env or 'wild_rescue' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, seed=self.args.env_args['seed'])
        else:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        # For record the action selections of each agent under different clusters
        # self.actions_batch = [{str(i): [0 for a in range(self.args.n_actions)] for i in range(self.args.n_skills)} for j in range(self.args.n_agents)]

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        # Only for record
        # selected_clusters = []
        # selected_actions = []

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, selected_skills = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # selected_clusters.append(selected_skills)
            # selected_actions.append(actions)

            # for i in range(self.args.n_agents):
            #     # For each agent i, 统计其在不同cluster下选择actions的次数
            #     # if actions[0][i].numpy() != 0:
            #     self.actions_batch[i][str(selected_skills[0][i].numpy())][actions[0][i].numpy()] += 1

            reward, terminated, env_info = self.env.step(actions[0])

            # print(self.t, selected_skills)
            # time.sleep(0.15)

            # self.env.render() #rware

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "skills": selected_skills,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # print("selected clusters", selected_clusters)
        # print("selected actions", selected_actions)

        # Select actions in the last stored state
        actions, selected_skills = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions, "skills": selected_skills}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        # print(prefix + "return_mean", np.mean(returns), len(returns), returns)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()