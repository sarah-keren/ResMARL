import numpy as np
from gym.spaces import Box, Dict

from social_dilemmas.envs.map_env import MapEnv


class MapEnvWithMessagesAndSelfRewardPrediction(MapEnv):
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
        self.use_messages_attribute = use_messages_attribute
        self.max_reward_value = None

    @property
    def observation_space(self):
        obs_space = super().observation_space.spaces
        # todo - when implementing mandatory - change the shape of the messages (like obs)
        if self.use_messages_attribute:
            obs_space = {
                **obs_space,
                "other_agent_messages": Box(
                    low=0, high=len(self.all_actions), shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "other_agent_predicted_rewards": Box(
                    low=0, high=self.max_reward_value, shape=(self.num_agents - 1,),
                    dtype=np.float32,
                ),
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
        predicted_rewards_for_current_step = {}

        if self.use_messages_attribute:
            messages = {agent_id: extended_action[1] for agent_id, extended_action in
                        actions.items()}
            predicted_rewards_for_current_step = {agent_id: extended_action[-1][0] for agent_id, extended_action in
                                                  actions.items()}
            actions = {agent_id: extended_action[0] for agent_id, extended_action in
                       actions.items()}

        observations, rewards, dones, info = super().step(actions)

        if self.use_messages_attribute:
            for agent in self.agents.values():
                # TODO improve complexity by list comprehension over a the whole messages array
                others_messages = np.array(
                    [messages[key] for key in sorted(messages.keys()) if key != agent.agent_id]
                ).astype(np.uint8)
                others_predicted_rewards_for_current_step = np.array(
                    [np.clip(predicted_rewards_for_current_step[key], a_min=0.1,
                             a_max=self.max_reward_value) for key in
                     sorted(predicted_rewards_for_current_step.keys()) if
                     key != agent.agent_id]
                ).astype(np.float32)
                observations[agent.agent_id] = {
                    **observations[agent.agent_id],
                    "other_agent_messages": others_messages,
                    "other_agent_predicted_rewards": others_predicted_rewards_for_current_step,
                }

        return observations, rewards, dones, info

    def reset(self):
        """
        Also need overwrite since it returns observation and messages
        """
        observations = super().reset()
        if self.use_messages_attribute:
            others_messages = np.array([0 for _ in range(self.num_agents - 1)]).astype(np.uint8)
            others_predicted_rewards_for_current_step = np.array([0 for _ in range(self.num_agents - 1)]).astype(
                np.float32)
            for agent in self.agents.values():
                observations[agent.agent_id] = {
                    **observations[agent.agent_id],
                    "other_agent_messages": others_messages,
                    "other_agent_predicted_rewards": others_predicted_rewards_for_current_step,
                }

        return observations