import random

import numpy as np
from gym.spaces import Tuple, Box
from numpy.random import rand

from config.constants import HARVEST_BASE_ACTION_SPACE_SIZE, REWARD_UPPER_BOUND, HARVEST_PERIODIC_PERT_FREQUENCY, \
    MAX_PERTURBATION_MAGNITUDE, MAX_PERCENT_OF_WALLS_ALLOWED
from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.envs.map_env_with_mandatory_messages import MapEnvWithMandatoryMessages
from social_dilemmas.envs.map_env_with_messages_global_confusion import MapEnvWithMessagesAndGlobalRewardPrediction
from social_dilemmas.envs.map_env_with_messages_self_confusion import MapEnvWithMessagesAndSelfRewardPrediction
from social_dilemmas.maps import HARVEST_MAP

APPLE_RADIUS = 2

# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

HARVEST_VIEW_SIZE = 7


# harvest with mandatory messages and perturbations

class HarvestPerturbationsEnvWithMessagesMandatory(MapEnvWithMandatoryMessages):
    def __init__(
            self,
            ascii_map=HARVEST_MAP,
            num_agents=1,
            return_agent_actions=False,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = HARVEST_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return Tuple([
            DiscreteWithDType(HARVEST_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            DiscreteWithDType(1, dtype=np.uint8),
        ])

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(), agent.get_orientation(), self.all_actions["FIRE"], fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        removed_walls = []
        new_walls = []
        removed_apples = []
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            # apply walls and apples perturbations
            # remove previously added walls
            removed_walls = [(wall_row, wall_col, b' ') for wall_row, wall_col, _ in self.previously_added_walls]
            # add new walls
            new_walls = self.perturb_walls()
            self.previously_added_walls = new_walls

            # remove some of existing apples
            removed_apples = self.perturb_existing_apples()
            # reduce amount of new apples
            new_apples = self.perturb_spawned_apples(new_apples)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples

        self.update_map(updates_to_map)

    def perturb_walls(self):
        template_map = self.get_map_with_agents()
        empty_spaces = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b' ':
                    empty_spaces.append((i, j))

        amount_of_walls_to_add = int(self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
            empty_spaces))

        new_walls = random.sample(empty_spaces, amount_of_walls_to_add)
        new_walls = [(row, col, b'@') for row, col in new_walls]
        self.perturb_map_colors()

        return new_walls

    def perturb_existing_apples(self):
        template_map = self.get_map_with_agents()
        apples_coordinates = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b'A':
                    apples_coordinates.append((i, j))

        amount_of_apples_to_remove = int(
            self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
                apples_coordinates))

        new_walls = random.sample(apples_coordinates, amount_of_apples_to_remove)
        new_walls = [(row, col, b' ') for row, col in new_walls]

        return new_walls

    def perturb_spawned_apples(self, new_apples):
        percentage_of_apples_to_keep = 1 - self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED
        amount_of_apples_to_keep = int(percentage_of_apples_to_keep * len(new_apples))

        new_apples = random.sample(new_apples, amount_of_apples_to_keep)

        return new_apples

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                    0 <= x + j < self.world_map.shape[0]
                                    and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# harvest with global confusion and perturbations

class HarvestPerturbationsEnvWithMessagesGlobal(MapEnvWithMessagesAndGlobalRewardPrediction):
    def __init__(
            self,
            ascii_map=HARVEST_MAP,
            num_agents=1,
            return_agent_actions=False,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = HARVEST_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return Tuple([
            DiscreteWithDType(HARVEST_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            DiscreteWithDType(HARVEST_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            Box(low=-self.max_reward_value, high=self.max_reward_value, shape=(1,), dtype=np.float32)
        ])

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(), agent.get_orientation(), self.all_actions["FIRE"], fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        removed_walls = []
        new_walls = []
        removed_apples = []
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            # apply walls and apples perturbations
            # remove previously added walls
            removed_walls = [(wall_row, wall_col, b' ') for wall_row, wall_col, _ in self.previously_added_walls]
            # add new walls
            new_walls = self.perturb_walls()
            self.previously_added_walls = new_walls

            # remove some of existing apples
            removed_apples = self.perturb_existing_apples()
            # reduce amount of new apples
            new_apples = self.perturb_spawned_apples(new_apples)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples

        self.update_map(updates_to_map)

    def perturb_walls(self):
        template_map = self.get_map_with_agents()
        empty_spaces = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b' ':
                    empty_spaces.append((i, j))

        amount_of_walls_to_add = int(self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
            empty_spaces))

        new_walls = random.sample(empty_spaces, amount_of_walls_to_add)
        new_walls = [(row, col, b'@') for row, col in new_walls]
        self.perturb_map_colors()

        return new_walls

    def perturb_existing_apples(self):
        template_map = self.get_map_with_agents()
        apples_coordinates = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b'A':
                    apples_coordinates.append((i, j))

        amount_of_apples_to_remove = int(
            self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
                apples_coordinates))

        new_walls = random.sample(apples_coordinates, amount_of_apples_to_remove)
        new_walls = [(row, col, b' ') for row, col in new_walls]

        return new_walls

    def perturb_spawned_apples(self, new_apples):
        percentage_of_apples_to_keep = 1 - self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED
        amount_of_apples_to_keep = int(percentage_of_apples_to_keep * len(new_apples))

        new_apples = random.sample(new_apples, amount_of_apples_to_keep)

        return new_apples

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                    0 <= x + j < self.world_map.shape[0]
                                    and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# harvest with self confusion and perturbations

class HarvestPerturbationsEnvWithMessagesSelf(MapEnvWithMessagesAndSelfRewardPrediction):
    def __init__(
            self,
            ascii_map=HARVEST_MAP,
            num_agents=1,
            return_agent_actions=False,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = HARVEST_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return Tuple([
            DiscreteWithDType(HARVEST_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            DiscreteWithDType(HARVEST_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            Box(low=-self.max_reward_value, high=self.max_reward_value, shape=(1,), dtype=np.float32)
        ])

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(), agent.get_orientation(), self.all_actions["FIRE"], fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        removed_walls = []
        new_walls = []
        removed_apples = []
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            # apply walls and apples perturbations
            # remove previously added walls
            removed_walls = [(wall_row, wall_col, b' ') for wall_row, wall_col, _ in self.previously_added_walls]
            # add new walls
            new_walls = self.perturb_walls()
            self.previously_added_walls = new_walls

            # remove some of existing apples
            removed_apples = self.perturb_existing_apples()
            # reduce amount of new apples
            new_apples = self.perturb_spawned_apples(new_apples)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples

        self.update_map(updates_to_map)

    def perturb_walls(self):
        template_map = self.get_map_with_agents()
        empty_spaces = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b' ':
                    empty_spaces.append((i, j))

        amount_of_walls_to_add = int(self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
            empty_spaces))

        new_walls = random.sample(empty_spaces, amount_of_walls_to_add)
        new_walls = [(row, col, b'@') for row, col in new_walls]
        self.perturb_map_colors()

        return new_walls

    def perturb_existing_apples(self):
        template_map = self.get_map_with_agents()
        apples_coordinates = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b'A':
                    apples_coordinates.append((i, j))

        amount_of_apples_to_remove = int(
            self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
                apples_coordinates))

        new_walls = random.sample(apples_coordinates, amount_of_apples_to_remove)
        new_walls = [(row, col, b' ') for row, col in new_walls]

        return new_walls

    def perturb_spawned_apples(self, new_apples):
        percentage_of_apples_to_keep = 1 - self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED
        amount_of_apples_to_keep = int(percentage_of_apples_to_keep * len(new_apples))

        new_apples = random.sample(new_apples, amount_of_apples_to_keep)

        return new_apples

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                    0 <= x + j < self.world_map.shape[0]
                                    and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# classic harvest with perturbations

class HarvestPerturbationEnv(MapEnv):
    def __init__(
            self,
            ascii_map=HARVEST_MAP,
            num_agents=1,
            return_agent_actions=False,
            use_collective_reward=False,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        # attributes for perturbations
        self.perturbations_frequency = HARVEST_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(), agent.get_orientation(), self.all_actions["FIRE"], fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        removed_walls = []
        new_walls = []
        removed_apples = []
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            # apply walls and apples perturbations
            # remove previously added walls
            removed_walls = [(wall_row, wall_col, b' ') for wall_row, wall_col, _ in self.previously_added_walls]
            # add new walls
            new_walls = self.perturb_walls()
            self.previously_added_walls = new_walls

            # remove some of existing apples
            removed_apples = self.perturb_existing_apples()
            # reduce amount of new apples
            new_apples = self.perturb_spawned_apples(new_apples)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples

        self.update_map(updates_to_map)

    def perturb_walls(self):
        template_map = self.get_map_with_agents()
        empty_spaces = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b' ':
                    empty_spaces.append((i, j))

        amount_of_walls_to_add = int(self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
            empty_spaces))

        new_walls = random.sample(empty_spaces, amount_of_walls_to_add)
        new_walls = [(row, col, b'@') for row, col in new_walls]

        self.perturb_map_colors()

        return new_walls

    def perturb_existing_apples(self):
        template_map = self.get_map_with_agents()
        apples_coordinates = []
        for i in range(template_map.shape[0]):
            for j in range(template_map.shape[-1]):
                if template_map[i, j] == b'A':
                    apples_coordinates.append((i, j))

        amount_of_apples_to_remove = int(
            self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED * len(
                apples_coordinates))

        new_walls = random.sample(apples_coordinates, amount_of_apples_to_remove)
        new_walls = [(row, col, b' ') for row, col in new_walls]

        return new_walls

    def perturb_spawned_apples(self, new_apples):
        percentage_of_apples_to_keep = 1 - self.perturbations_magnitude_relative_to_max * MAX_PERCENT_OF_WALLS_ALLOWED
        amount_of_apples_to_keep = int(percentage_of_apples_to_keep * len(new_apples))

        new_apples = random.sample(new_apples, amount_of_apples_to_keep)

        return new_apples

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                    0 <= x + j < self.world_map.shape[0]
                                    and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255
