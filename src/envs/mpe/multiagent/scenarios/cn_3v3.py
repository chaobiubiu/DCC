import numpy as np
from src.envs.mpe.multiagent.core import World, Agent, Landmark
from src.envs.mpe.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True
        # Flag
        self.step_pass = False
        # Task params
        self.num_agents = 3
        self.num_landmarks = 3

        self.global_states = None
        self.obs_mask = None
        self.sight_range = 99999
        self.agent_size = 0.15

        self.local_state_size = 16  # [vel+pos+ralative pos to all landmarks+relative pos to all agents], 2+2+2x3+2x3=16
        self.state_size = int(self.local_state_size * (self.num_agents + self.num_landmarks))
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
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # extra params
        self.dis_matrix = np.zeros((self.num_landmarks, self.num_agents))
        self.collision_matrix = np.zeros((self.num_agents, self.num_agents))

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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in range(self.num_landmarks):
            rew -= min(self.dis_matrix[l])
        if agent.collide:
            rew -= np.sum(self.collision_matrix[agent.index])       # 这里需要将当前agent.index对应的那一行的碰撞数量累加
        return rew

    def get_states_masks(self, world):
        '''Note: self.step_pass should be reset to False in the main step function when the current timesteps passes'''
        if not self.step_pass:
            landmarks_pos = np.array([l.state.p_pos for l in world.landmarks])  # len(landmarks_pos) = self.num_landmark
            landmarks_pos_rep = np.repeat(landmarks_pos, self.num_agents, axis=0)  # 每个landmark的pos都复制了num_agent份
            agents_pos = np.array([a.state.p_pos for a in world.agents])  # len(agents_pos) = self.num_agent
            concat_agents_pos = np.concatenate(agents_pos, axis=0)
            agents_pos_rep = np.repeat([concat_agents_pos], self.num_landmarks, axis=0).reshape(self.num_landmarks * self.num_agents, -1)
            # 这里就可以计算每个landmark距离其他所有agent的距离
            dis_array = np.sqrt(np.sum(np.square(landmarks_pos_rep - agents_pos_rep), axis=1, keepdims=True))
            self.dis_matrix = dis_array.reshape(self.num_landmarks, self.num_agents)  # 每一元素代表当前行相应landmark距离当前列相应agent的distance.

            # calculate the distance between any two agents
            aux_agents_pos_rep = np.repeat(agents_pos, self.num_agents, axis=0)  # 每个agent的pos都复制了num_agents份
            # 由于self.num_landmark==self.num_agent，所以这里直接使用agents_pos_rep，相当于将所有agent的pos array整体复制num_landmark or num_agent次
            agent_dis_array = np.sqrt(np.sum(np.square(aux_agents_pos_rep - agents_pos_rep), axis=1, keepdims=True))
            self.agent_dis_matrix = agent_dis_array.reshape(self.num_agents, self.num_agents)  # 每一行代表当前行agent距离其他所有agent的距离
            self.collision_matrix = (self.agent_dis_matrix < self.agent_size * 2)  # If true, two agents collide.
            self.step_pass = True  # 该if语句内部的计算在每一时间步仅计算一次

            # 全局状态包括所有entity(agent and landmarks)的local states
            global_states = []
            for i, a in enumerate(world.agents):
                local_state = []
                local_state.append(a.state.p_pos)  # dim 2, pos
                local_state.append(a.state.p_vel)  # dim 2, vel
                for k, other in enumerate(world.agents):  # dim 2 * (num_agents)
                    local_state.append(other.state.p_pos - a.state.p_pos)
                for n, other_l in enumerate(world.landmarks):  # dim 2 * (max_num_landmarks)
                    local_state.append(other_l.state.p_pos - a.state.p_pos)
                local_state = np.concatenate(local_state, axis=0)
                global_states.append(local_state)

            for j, l in enumerate(world.landmarks):
                local_state = []
                local_state.append(l.state.p_pos)  # dim 2
                local_state.append(l.state.p_vel)  # dim 2
                for k, other in enumerate(world.agents):  # dim 2 * (max_num_agents)
                    local_state.append(other.state.p_pos - l.state.p_pos)
                for n, other_l in enumerate(world.landmarks):  # dim 2 * (max_num_landmarks)
                    local_state.append(other_l.state.p_pos - l.state.p_pos)
                local_state = np.concatenate(local_state, axis=0)
                global_states.append(local_state)

            self.global_states = global_states  # len(global_states)=num_agents+num_landmarks

            # In summary, calculate the partially observable masks
            # 根据距离计算所有agent/landmark间的距离矩阵
            agents_landmarks_pos = np.vstack((agents_pos, landmarks_pos))  # len()=self.num_agent+self.num_landmark
            agents_landmarks_pos_rep = np.repeat(agents_landmarks_pos, (self.num_agents + self.num_landmarks), axis=0)  # 其中以每一项为基本单位复制(num_agent+num_landmark)次
            aux_pos = np.concatenate(agents_landmarks_pos, axis=0)
            aux_pos_rep = np.repeat([aux_pos], (self.num_agents + self.num_landmarks), axis=0).reshape((self.num_agents + self.num_landmarks) ** 2, -1)  # 以整体为基本单位复制(num_agent+num_landmark)次
            aux_dis_array = np.sqrt(np.sum(np.square(agents_landmarks_pos_rep - aux_pos_rep), axis=1, keepdims=True))
            all_dis_matrix = aux_dis_array.reshape((self.num_agents + self.num_landmarks), (self.num_agents + self.num_landmarks))
            obs_mask = (all_dis_matrix > self.sight_range).astype(np.uint8)  # unobservable 1, observable 0
            # obs_mask记录了部分可观测的mask
            obs_mask = np.expand_dims(obs_mask, axis=-1)  # shape=(num_agents+num_landmarks, num_agents+num_landmarks, 1)
            self.obs_mask = obs_mask

        return self.global_states, self.obs_mask

    def observation(self, agent, world):
        # 根据距离矩阵考虑部分可观测的mask，然后直接对全局状态进行mask操作
        # 全局状态包括所有entity(agent and landmarks)的local states
        global_states = []
        for i, a in enumerate(world.agents):
            local_state = []
            local_state.append(a.state.p_pos)  # dim 2, pos
            local_state.append(a.state.p_vel)  # dim 2, vel
            for k, other in enumerate(world.agents):  # dim 2 * (max_num_agents)
                local_state.append(other.state.p_pos - a.state.p_pos)
            for n, other_l in enumerate(world.landmarks):  # dim 2 * (max_num_landmarks)
                local_state.append(other_l.state.p_pos - a.state.p_pos)
            local_state = np.concatenate(local_state, axis=0)
            global_states.append(local_state)

        for j, l in enumerate(world.landmarks):
            local_state = []
            local_state.append(l.state.p_pos)  # dim 2
            local_state.append(l.state.p_vel)  # dim 2
            for k, other in enumerate(world.agents):  # dim 2 * (max_num_agents)
                local_state.append(other.state.p_pos - l.state.p_pos)
            for n, other_l in enumerate(world.landmarks):  # dim 2 * (max_num_landmarks)
                local_state.append(other_l.state.p_pos - l.state.p_pos)
            local_state = np.concatenate(local_state, axis=0)
            global_states.append(local_state)

        return global_states