import numpy as np
from gym.spaces import Box, Dict, Tuple

from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv


class MapEnvWithMandatoryMessages(MapEnv):
    def __init__(
            self,
            ascii_map,
            extra_actions,
            view_len,
            num_agents=1,
            color_map=None,
            return_agent_actions=False,
            use_messages_attribute=False,
            use_collective_reward=False,
    ):
        super().__init__(ascii_map=ascii_map,
                         extra_actions=extra_actions,
                         view_len=view_len,
                         num_agents=num_agents,
                         color_map=color_map,
                         return_agent_actions=return_agent_actions,
                         use_collective_reward=use_collective_reward)
        self.agents_last_couple_raw_observation = {}
        self.agents_previous_action = {}
        self.agents_previous_reward = {}
        self.use_messages_attribute = use_messages_attribute

        for agent in self.agents.values():
            self.agents_last_couple_raw_observation[agent.agent_id] = []

    @property
    def observation_space(self):
        self.num_agents = len(list(self.agents.keys()))
        obs_space = super().observation_space.spaces
        if self.use_messages_attribute:
            obs_space = {
                **obs_space,
                "mandatory_broadcast_transitions": Tuple(
                    [obs_space['curr_obs'],
                     Box(
                         low=-100, high=100, shape=(self.num_agents - 1,), dtype=np.int64,
                     ),
                     Box(
                         low=-100, high=100, shape=(self.num_agents - 1,), dtype=np.int64,
                     ),
                     obs_space['curr_obs']
                     ] * (self.num_agents - 1)
                ),
                # "valuable_messages_indices": Box(low=0, high=1, shape=(self.num_agents - 1,), dtype=np.uint8),
            }
        obs_space = Dict(obs_space)
        # Change dtype so that ray can put all observations into one flat batch
        # with the correct dtype.
        # See DictFlatteningPreprocessor in ray/rllib/models/preprocessors.py.
        obs_space.dtype = np.uint8
        return obs_space

    def step(self, actions):
        """
        overwriting original function to take action and write down messages, different
        is that in current case actions are list of action and message.
        """
        self.beam_pos = []
        messages = {}

        if self.use_messages_attribute:
            # messages = {agent_id: extended_action[1] for agent_id, extended_action in
            #             actions.items()}
            # actions = {agent_id: extended_action[0] for agent_id, extended_action in
            #            actions.items()}
            actions = actions

        observations, rewards, dones, info = super().step(actions)

        if self.use_messages_attribute:
            for agent in self.agents.values():
                broadcast_transitions_tuples, valuable_transitions = self.derive_transitions_to_broadcast(messages,
                                                                                                          observations,
                                                                                                          agent)
                observations[agent.agent_id]["curr_obs"] = observations[agent.agent_id]["curr_obs"]
                observations[agent.agent_id]["mandatory_broadcast_transitions"] = tuple(broadcast_transitions_tuples)
                # observations[agent.agent_id]["valuable_messages_indices"] = valuable_transitions

        for agent in self.agents.values():
            if len(self.agents_last_couple_raw_observation[agent.agent_id]) == 2:
                self.agents_last_couple_raw_observation[agent.agent_id].pop()
            self.agents_last_couple_raw_observation[agent.agent_id].append(observations[agent.agent_id]["curr_obs"])
            self.agents_previous_action[agent.agent_id] = actions[agent.agent_id]
            self.agents_previous_reward[agent.agent_id] = rewards[agent.agent_id]

        return observations, rewards, dones, info

    def derive_transitions_to_broadcast(self, messages, observations, agent_recv):
        default_transitions_to_broadcast = []
        default_transitions_to_broadcast += [observations[agent_recv.agent_id]['curr_obs'],
                                             np.array([1]),
                                             np.array([-self.max_reward_value]),
                                             observations[agent_recv.agent_id]['curr_obs']
                                             ] * (self.num_agents - 1)
        tmp_agent = list(self.agents.values())[0]
        if len(self.agents_last_couple_raw_observation[tmp_agent.agent_id]) < 2 or tmp_agent not in list(
                self.agents_previous_action.keys()):
            return tuple(default_transitions_to_broadcast), np.array([0] * (self.num_agents - 1))

        transitions_to_broadcast = []
        valuable_indices = []
        for agent in self.agents.values():
            if agent.agent_id != agent_recv.agent_id:
                transitions_to_broadcast += [self.agents_last_couple_raw_observation[agent.agent_id][0],
                                             self.agents_previous_action[agent.agent_id],
                                             np.array(self.agents_previous_reward[agent.agent_id]),
                                             self.agents_last_couple_raw_observation[agent.agent_id][-1]]

            # if messages[agent.agent_id] == 1 and agent_recv.agent_id != agent.agent_id:
            #     valuable_indices.append(1)
            # else:
            #     valuable_indices.append(0)

        return tuple(np.array(transitions_to_broadcast))  # , np.array(valuable_indices)

    def reset(self):
        """
        Also need overwrite since it returns observation and messages
        """
        observations = super().reset()
        if self.use_messages_attribute:
            for agent in self.agents.values():
                observations[agent.agent_id] = {
                    **observations[agent.agent_id],
                    "mandatory_broadcast_transitions": tuple(np.array([
                                                                          [observations[agent.agent_id]['curr_obs'],
                                                                           np.array([1]),
                                                                           np.array([0]),
                                                                           observations[agent.agent_id]['curr_obs']
                                                                           ]] * (self.num_agents - 1)).reshape(-1))
                    ,
                    # "valuable_messages_indices": np.array([0] * (self.num_agents - 1)),
                }
        for agent in self.agents.values():
            self.agents_last_couple_raw_observation[agent.agent_id].append(observations[agent.agent_id]["curr_obs"])
        return observations
