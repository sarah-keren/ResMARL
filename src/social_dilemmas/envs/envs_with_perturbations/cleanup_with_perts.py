import random

import numpy as np
from gym.spaces import Tuple, Box
from numpy.random import rand

from config.constants import CLEANUP_BASE_ACTION_SPACE_SIZE, REWARD_UPPER_BOUND, CLEANUP_PERIODIC_PERT_FREQUENCY, \
    MAX_PERTURBATION_MAGNITUDE, MAX_PERCENT_OF_WALLS_ALLOWED
from social_dilemmas.envs.agent import CleanupAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.envs.map_env_with_mandatory_messages import MapEnvWithMandatoryMessages
from social_dilemmas.envs.map_env_with_messages_global_confusion import MapEnvWithMessagesAndGlobalRewardPrediction
from social_dilemmas.envs.map_env_with_messages_self_confusion import MapEnvWithMessagesAndSelfRewardPrediction
from social_dilemmas.maps import CLEANUP_MAP

# Add custom actions to the agent
_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}  # length of firing beam, length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    b"S": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue stream cell
    b"H": np.array([99, 156, 194], dtype=np.uint8),  # Brown waste cells
    b"R": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue river cell
}

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


# cleanup with mandatory messages

class CleanupPerturbationsEnvWithMessagesMandatory(MapEnvWithMandatoryMessages):
    def __init__(
            self,
            ascii_map=CLEANUP_MAP,
            num_agents=1,
            return_agent_actions=True,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)
        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = CLEANUP_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8)

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""

        self.compute_probabilities()

        # spawn the apples and waste
        new_apples_and_waste = self.spawn_apples_and_waste()
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
            new_apples_and_waste = self.perturb_spawned_apples(new_apples_and_waste)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples_and_waste

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

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                                     1
                                     - (waste_density - thresholdRestoration)
                                     / (thresholdDepletion - thresholdRestoration)
                             ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# cleanup with global confusion and perturbations

class CleanupPerturbationsEnvWithMessagesGlobal(MapEnvWithMessagesAndGlobalRewardPrediction):
    def __init__(
            self,
            ascii_map=CLEANUP_MAP,
            num_agents=1,
            return_agent_actions=True,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)
        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = CLEANUP_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        # base_action_space = DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8)
        #
        # base_action_space = Tuple([base_action_space] * 2) if self.use_messages_attribute else base_action_space
        # return base_action_space + Box(low=0, high=self.max_confusion_value, shape=(1, ), dtype=np.float32)
        return Tuple([
            DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
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

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""

        self.compute_probabilities()

        # spawn the apples and waste
        new_apples_and_waste = self.spawn_apples_and_waste()
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
            new_apples_and_waste = self.perturb_spawned_apples(new_apples_and_waste)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples_and_waste

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

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                                     1
                                     - (waste_density - thresholdRestoration)
                                     / (thresholdDepletion - thresholdRestoration)
                             ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# cleanup with self conf properties and perturbations

class CleanupPerturbationsEnvWithMessagesSelf(MapEnvWithMessagesAndSelfRewardPrediction):
    def __init__(
            self,
            ascii_map=CLEANUP_MAP,
            num_agents=1,
            return_agent_actions=True,
            use_collective_reward=False,
            use_messages_attribute=True,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            use_messages_attribute=use_messages_attribute
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)
        self.max_reward_value = REWARD_UPPER_BOUND

        # attributes for perturbations
        self.perturbations_frequency = CLEANUP_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        # base_action_space = DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8)
        #
        # base_action_space = Tuple([base_action_space] * 2) if self.use_messages_attribute else base_action_space
        # return base_action_space + Box(low=0, high=self.max_confusion_value, shape=(1, ), dtype=np.float32)
        return Tuple([
            DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
            DiscreteWithDType(CLEANUP_BASE_ACTION_SPACE_SIZE, dtype=np.uint8),
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

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()

        # spawn the apples and waste
        new_apples_and_waste = self.spawn_apples_and_waste()
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
            new_apples_and_waste = self.perturb_spawned_apples(new_apples_and_waste)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples_and_waste

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

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                                     1
                                     - (waste_density - thresholdRestoration)
                                     / (thresholdDepletion - thresholdRestoration)
                             ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255


# classic cleanup env and perturbations

class CleanupPerturbationsEnv(MapEnv):
    def __init__(
            self,
            ascii_map=CLEANUP_MAP,
            num_agents=1,
            return_agent_actions=False,
            use_collective_reward=False,
            perturbation_magnitude=50,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

        # attributes for perturbations
        self.perturbations_frequency = CLEANUP_PERIODIC_PERT_FREQUENCY
        self.perturbations_magnitude = perturbation_magnitude
        self.perturbations_magnitude_relative_to_max = (self.perturbations_magnitude / MAX_PERTURBATION_MAGNITUDE)
        self.time_step_in_instance = 0
        self.previously_added_walls = []

    @property
    def action_space(self):
        return DiscreteWithDType(9, dtype=np.uint8)

    def step(self, actions):
        self.time_step_in_instance += 1
        if not self.time_step_in_instance % self.perturbations_frequency and self.time_step_in_instance:
            self.reset(reset_time_steps=False)
        return super().step(actions)

    def reset(self, reset_time_steps=True):
        if reset_time_steps:
            self.time_step_in_instance = 0
        return super().reset()

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()

        # spawn the apples and waste
        new_apples_and_waste = self.spawn_apples_and_waste()
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
            new_apples_and_waste = self.perturb_spawned_apples(new_apples_and_waste)

        updates_to_map = removed_walls + removed_apples + new_walls + new_apples_and_waste

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

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                                     1
                                     - (waste_density - thresholdRestoration)
                                     / (thresholdDepletion - thresholdRestoration)
                             ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def perturb_map_colors(self):
        self.color_map[''][0] = (self.color_map[''][0] + 50) % 255
        self.color_map['@'][0] = (self.color_map['@'][0] + 50) % 255
