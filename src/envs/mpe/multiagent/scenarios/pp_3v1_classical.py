import numpy as np
from src.envs.mpe.multiagent.core import World, Agent, Landmark
from src.envs.mpe.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True

        self.num_predators = 3
        self.num_preys = 1
        self.num_agents = self.num_predators
        self.num_total_agents = self.num_predators + self.num_preys
        self.num_landmarks = 2
        # the setting of dim
        self.predator_size = 0.075
        self.prey_size = 0.05
        # [vel+pos+relative pos to all landmarks+relative pos to agents -i + velocity of preys], 2+2+2X2+2X3+2x1=16
        self.local_state_size = 16
        self.state_size = int(self.local_state_size * self.num_predators)
        # the episode length
        self.episode_limit = 25

        # Add agents
        world.agents = [Agent() for i in range(self.num_total_agents)]
        for i, agent in enumerate(world.agents):
            agent.index = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < self.num_predators else False
            agent.size = self.predator_size if agent.adversary else self.prey_size
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            # Design scripted rule for the preys.
            agent.action_callback = None if agent.adversary else self.prey_policy
            agent.view_radius = 0.25
            print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        # Add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        self.reset_world(world)
        self.score_function = "sum"
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def prey_policy(self, agent, world):
        action = None
        n = 100 # number of positions sampled
        # sample actions randomly from a target cycle
        length = np.sqrt(np.random.uniform(0, 1, n))
        angle = np.pi * np.random.uniform(0, 2, n)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        # evaluate score for each position
        # check whether positions are reachable
        # sample a few evenly spaced points on the way and see if they collide with anything
        scores  = np.zeros(n, dtype=np.float32)

        if self.score_function == 'min':
            relative_dis = []
            adv_names = []
            adversaries = self.adversaries(world)
            proj_pos = np.vstack((x, y)).transpose() + agent.state.p_pos    # 这里是直接考虑执行action后的target position，不像上面考虑路途中
            for adv in adversaries:
                relative_dis.append(np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))))
                adv_names.append(adv.name)
            min_dis_adv_name = adv_names[np.argmin(relative_dis)]    # 距离prey 当前位置 最近的predator
            for adv in adversaries:
                delta_pos = adv.state.p_pos - proj_pos
                dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
                dist_min = adv.size + agent.size
                scores[dist < dist_min] = -9999999      # 如果在target_pos处与adversary发生碰撞，则将该处分数设置为-9999999
                if adv.name == min_dis_adv_name:
                    scores += dist                  # 这里是只考虑距离同一个predator的dist

        elif self.score_function == 'sum':
            cos_, sin_ = np.cos(angle), np.sin(angle)
            n_iter = 5
            for i in range(n_iter):     # 这里是判断路途中是否会发生碰撞，将执行action后到达的target_pos距离当前prey_pos的路段分成n段来考虑
                waypoints_length = (length / float(n_iter)) * (i + 1)
                x_wp = waypoints_length * cos_
                y_wp = waypoints_length * sin_
                proj_pos = np.vstack((x_wp, y_wp)).transpose() + agent.state.p_pos  # 这个粗略计算选择action后的target pos
                for _agent in world.agents:
                    if _agent.adversary:        # predator
                        delta_pos = _agent.state.p_pos - proj_pos
                        dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
                        dist_min = _agent.size + agent.size
                        scores[dist < dist_min] = -9999999      # 如果agent到达中间子target_pos的路途中与predator发生collide
                        if i == (n_iter - 1) and _agent.movable:
                            scores += dist          # 这里判断是(n_iter-1)次循环，最终将所有n种action到达的target_pos与prey当前位置的距离加进score
        else:
            raise Exception('Unknown score function {}'.format(self.score_function))

        # move to the best position
        best_idx = np.argmax(scores)        # 如果执行某一action的路途中与predator发生collide，则对应分数就会非常低
        chosen_action = np.array([x[best_idx], y[best_idx]], dtype=np.float32)      # 这里是两维agent.action.u，是continuous action setting，在world.step时需要注意一下
        if scores[best_idx] < 0:        # 如果n种action对应的路径上都已经被predator封锁，那prey就无法动弹了。
            chosen_action *= 0.0        # cannot go anywhere
        return chosen_action

    # def prey_policy(self, agent, world):
    #     prey_action = np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1])
    #     return prey_action

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     agents = self.good_agents(world)
    #     if agent.collide:
    #         for ag in agents:
    #             if self.is_collision(ag, agent):
    #                 rew += 10
    #     return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:          # 2*num_landmarks
            dist = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if not entity.boundary and (agent.view_radius >= 0) and dist <= agent.view_radius:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(np.array([0., 0.]))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)     # 2*(num_agents-1)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)                     # 2*1
            else:
                other_pos.append(np.array([0., 0.]))
                if not other.adversary:
                    other_vel.append(np.array([0., 0.]))
        # 2+2+2*(num_landmarks:2)+2*(num_agents-1)+2*1=16
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)