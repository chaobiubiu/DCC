import numpy as np
from src.envs.mpe.multiagent.core import World, Agent, Landmark
from src.envs.mpe.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        print("Use sparse cn_6v3.")
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True
        # Task params
        self.num_agents = 6
        self.num_landmarks = 3

        self.global_states = None
        self.sight_range = 99999
        self.agent_size = 0.15

        self.local_state_size = 30  # [vel+pos+relative pos to all landmarks+relative pos and comm to agents -i], 2+2+2x3+2x5+2x5=30
        self.state_size = int(self.local_state_size * self.num_agents)      # 30x6=180
        # the episode length
        self.episode_limit = 25
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.index = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.reached = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.index = i
            landmark.collide = False
            landmark.movable = False
        self.colors = np.array([[221,127,106],
                                [204,169,120],
                                [191,196,139],
                                [176,209,152],
                                [152,209,202],
                                [152,183,209],
                                [152,152,209],
                                [185,152,209],
                                [209,152,203],
                                [209,152,161]])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        goal_index = np.arange(len(world.landmarks))
        np.random.shuffle(goal_index)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # Randomly select one landmark as the target for per two agents
            # 0st, 1st agent have 0st landmark, 2st, 3st have 1st landmark, 4st, 5st have 2st landmark
            l_label = int(i // 2)
            # 0st, 1st agents have the same colors, also 2st, 3st agents and 4st, 5st agents are the same
            agent.color = self.colors[l_label] / 256
            agent.target_l = world.landmarks[goal_index[l_label]]
            # The target landmarks have the same colors as its followers.
            agent.target_l.color = self.colors[l_label] / 256

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         rew -= min(dists)
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #     return rew

    # def reward(self, agent, world):
    #     # Agents are rewarded incrementally when part of them reach their target landmarks respectively
    #     reach_array = [0.0 for _ in range(self.num_agents)]
    #     for i, agent in enumerate(world.agents):
    #         curr_target_l = agent.target_l
    #         curr_distance = np.sqrt(np.sum(np.square(agent.state.p_pos - curr_target_l.state.p_pos)))
    #         # If any single agent has reached its target landmark, receive 0.6 rewards.
    #         if curr_distance <= 0.05:
    #             reach_array[i] = 0.6
    #     rew = np.sum(reach_array)
    #     # In the environment_classical, the global rewards = individual rewards * n_agents
    #     rew = rew / self.num_agents
    #     return rew

    def reward(self, agent, world):
        # Agents are rewarded incrementally when part of them reach their target landmarks respectively
        reach_array = [0.0 for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            curr_target_l = agent.target_l
            curr_distance = np.sqrt(np.sum(np.square(agent.state.p_pos - curr_target_l.state.p_pos)))
            # If any single agent has reached its target landmark, receive 0.6 rewards.
            if curr_distance <= 0.05:
                reach_array[i] = 1.0
        rew_array = [0.0 for _ in range(self.num_landmarks)]
        for i in range(self.num_agents):
            label = int(i // 2)
            rew_array[label] += reach_array[i]
        # If per two agents both reach their target landmark, receive +1.0
        rew_array = (np.array(rew_array) == 2.0)
        # print(rew_array)
        rew = np.sum(rew_array)
        # In the environment_classical, the global rewards = individual rewards * n_agents
        # rew = rew / self.num_agents
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        target_l_pos = []
        other_l_pos = []
        for entity in world.landmarks:  # 2 * num_landmarks
            if entity.index == agent.target_l.index:
                target_l_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                other_l_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)              # 2*(num_agents-1)
            other_pos.append(other.state.p_pos - agent.state.p_pos)     # 2*(num_agents-1)
        # 2+2+2*num_landmarks+2*(num_agents-1)+2*(num_agents-1)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + target_l_pos + other_l_pos + other_pos + comm)

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)           # 2*num_landmarks
    #     # entity colors
    #     entity_color = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_color.append(entity.color)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)              # 2*(num_agents-1)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)     # 2*(num_agents-1)
    #     # 2+2+2*num_landmarks+2*(num_agents-1)+2*(num_agents-1)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)