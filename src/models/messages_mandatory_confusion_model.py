# still in progress
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.model import restore_original_dimensions, flatten
from ray.rllib.utils.annotations import PublicAPI

from config.constants import REWARD_UPPER_BOUND, EPSILON_CONSTANT
from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.global_next_reward_lstm import GlobalNextRewardPredictionsLSTM
from models.next_reward_lstm import NextRewardLSTM

tf = try_import_tf()


class MandatoryMessagesConfusionModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, adjusting_period=1e6):
        """
        A base model that uses messages to reduce its self confusion.

        :param obs_space: The observation space shape.
        :param action_space: The amount of available actions to this agent.
        :param num_outputs: The amount of available actions to this agent.
        :param model_config: The model config dict. Used to determine size of conv and fc layers.
        :param name: The model name.
        """
        super(MandatoryMessagesConfusionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self._other_agent_actions = None
        self._visibility = None

        self._actions_model_out = None
        self._actions_value_out = None
        self._value_out = None
        self._messages_model_out = None
        self._model_out = None
        self._messages_value_out = None
        self._true_one_hot_messages = None
        self._agent_prev_actions = None
        self._prev_value = None
        self.td_errors = []
        self.counter_of_seen_states = 0
        self.td_error_history = []
        self.adjusting_period = adjusting_period

        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.actions_num_outputs = num_outputs  # - 1
        self.current_td_errors = None

        self.num_other_agents = model_config["custom_options"]["num_other_agents"]
        self.influence_divergence_measure = model_config["custom_options"][
            "influence_divergence_measure"
        ]
        self.influence_only_when_visible = model_config["custom_options"][
            "influence_only_when_visible"
        ]
        self.train_moa_only_when_visible = model_config["custom_options"][
            "train_moa_only_when_visible"
        ]

        self.encoder_model = self.create_messages_model_encoder(obs_space, model_config)

        self.register_variables(self.encoder_model.variables)
        self.encoder_model.summary()

        inner_obs_space = self.encoder_model.output_shape[-1]
        # Action selection/value function
        cell_size = model_config["custom_options"].get("cell_size")

        # note that action space is [action, message, conf_level]
        self.actions_policy_model = ActorCriticLSTM(
            inner_obs_space,
            action_space,
            self.actions_num_outputs,
            model_config,
            "actions_policy",
            cell_size=cell_size,
        )

        self.register_variables(self.actions_policy_model.rnn_model.variables)
        self.actions_policy_model.rnn_model.summary()

    @staticmethod
    def create_messages_model_encoder(obs_space, model_config):
        """
        Creates the convolutional part of the mesages mode, has two output heads, one for the messages and one for the
        actions.
        Also casts the input uint8 observations to float32 and normalizes them to the range [0,1].
        :param obs_space: The agent's observation space.
        :param model_config: The config dict containing parameters for the convolution type/shape.
        :return: A new Model object containing the convolution.
        """
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # Determine vision network input shape: add an extra none for the time dimension
        inputs = tf.keras.layers.Input(shape=original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        if len(inputs.shape) > 2:
            last_layer = tf.keras.backend.cast(inputs, tf.float32)
            last_layer = tf.math.divide(last_layer, 255.0)  # todo: replace here with normalization constant
        else:
            last_layer = inputs

        # Build the CNN layers
        conv_out = build_conv_layers(model_config, last_layer)

        # Add the fully connected layers
        last_layer = build_fc_layers(model_config, conv_out, "actions_policy")

        return tf.keras.Model(inputs, [last_layer], name="Baseline_Encoder_Model")

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Evaluate the model.
        Adds time dimension to batch before sending inputs to forward_rnn()
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and state.
        """
        self.counter_of_seen_states += 1
        self._other_agent_actions = input_dict["obs"]["other_agent_actions"]
        self._visibility = input_dict["obs"]["visible_agents"]
        self._agent_prev_actions = input_dict["prev_actions"]
        self.sort_replay_buffer_with_confusion_measure(input_dict)

        ac_critic_encoded_obs = self.encoder_model(inputs=input_dict["obs"]["curr_obs"])
        rnn_input_dict = {
            "ac_trunk": ac_critic_encoded_obs,
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": input_dict["prev_actions"],
        }

        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        h1, c1, *_ = state.copy()
        self.current_td_errors, _ = self.get_td_error_on_recieved_transitions(input_dict, state.copy())

        output, new_state = self.forward_rnn(input_dict=rnn_input_dict,
                                             state=state,
                                             seq_lens=seq_lens)

        self._prev_value = self.value_function()

        return tf.reshape(output, [-1, self.num_outputs], name='reshape_output_forward'), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the LSTM.
        Implicitly assigns the value function output to self_value_out, and does not return this.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and new state.
        """
        # 1 - actions, 2 - messages, 3 - reward predictor
        h1, c1, *_ = state

        # Compute the next action
        ac_pass_dict = {
            "curr_obs": input_dict["ac_trunk"]
        }
        (self._actions_model_out, self._actions_value_out, output_h1,
         output_c1,) = self.actions_policy_model.forward_rnn(
            ac_pass_dict, [h1, c1], seq_lens
        )

        return self._actions_model_out, [output_h1, output_c1]

    def sort_replay_buffer_with_confusion_measure(self, input_dict):
        """
        We have the actual reward + we have the estimated reward given our vector of messages.
        So given some predicted R we can calculate the achieved confusion.
        We define the negative of the reward (of the messages) to be the inverse of:
        The dist(l1, l2...) between the minimal confusion and the actual confusion.
        """
        # td error: R_(t+1) + gamma*V(s_t+1) - V(s_t)
        # For each step we check if td-error is under some (maybe dynamic) threshold, and if so -
        # we broadcast the transition
        if self._prev_value is None:
            return

        v_current = tf.reduce_max(self.value_function())
        r_current = tf.reduce_max(input_dict["prev_rewards"])
        v_prev = tf.reduce_max(self._prev_value)

        td_error = r_current + 0.99 * v_current - v_prev

        self._messages_model_out = tf.cast(self.is_td_error_out_of_range(td_error), tf.int32)
        # axis=-1) #self._reshaped_as_one_hot(
        # tf.expand_dims(tf.cast(self.is_td_error_out_of_range(td_error), tf.int32),
        #                axis=-1), 2,
        # 'mandatory_message')

    def is_td_error_out_of_range(self, td_error: float):
        self.counter_of_seen_states += 1
        self.td_error_history.append(td_error)
        if self.counter_of_seen_states < self.adjusting_period:
            return False
        if self.counter_of_seen_states == self.adjusting_period:
            self.td_error_history = self.td_error_history[len(self.td_error_history) // 2:]

        mean_td_error, std_td_error = tf.reduce_mean(self.td_error_history), tf.math.reduce_std(self.td_error_history)

        return tf.where(td_error >= 0.6 * std_td_error, tf.ones_like(td_error), tf.zeros_like(td_error))

    @staticmethod
    def _reshaped_as_one_hot(raw_tensor, encoding_vector_length, name):
        """
        Converts the collection of vectors from a number encoding to a one-hot encoding.
        Then, flattens the one-hot encoding so that all concatenated one-hot vectors are the same
        dimension. E.g. with a num_outputs (action_space.n) of 3:
        _reshaped_one_hot_actions([0,1,2]) returns [1,0,0,0,1,0,0,0,1]
        :param raw_tensor: The tensor containing numeric encoding.
        :return: Tensor containing one-hot reshaped action values.
        """
        one_hot_encoding = tf.keras.backend.one_hot(indices=raw_tensor, num_classes=encoding_vector_length)

        # Extract partially known tensor shape and combine with "vector"_layer known shape
        # This combination is a bit contrived for a reason: the shape cannot be determined otherwise
        batch_time_dims = [
            tf.shape(one_hot_encoding)[k] for k in range(one_hot_encoding.shape.rank - 2)
        ]
        reshape_dims = batch_time_dims + [raw_tensor.shape[-1] * encoding_vector_length]
        reshaped = tf.reshape(one_hot_encoding, shape=reshape_dims, name=name)
        return reshaped

    def action_logits(self):
        """
        :return: The action logits from the latest forward pass.
        """
        return tf.reshape(self._actions_model_out, [-1])

    def value_function(self):
        """
        :return: The value function result from the latest forward pass.
        """
        return tf.reshape(self._actions_value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        """
        :return: Initial state of this model. This model only has LSTM state from the policy_model.
        """
        return self.actions_policy_model.get_initial_state()

    def agent_prev_actions(self):
        return self._agent_prev_actions

    def current_mandatory_td_errors(self):
        return self.current_td_errors

    def get_td_error_on_recieved_transitions(self, input_dict, state):
        # tf.zeros(shape=(1,), name='td_errors_from_messages')
        td_errors_sum = tf.zeros(shape=(1,))
        # 1. process hypothetical value for each state transmitted in the messages
        np_arrays_of_transitions = np.array_split(input_dict["obs"]["mandatory_broadcast_transitions"],
                                                  self.num_other_agents)
        raw_others_transitions = [list(transition) for transition in np_arrays_of_transitions]
        seq_lens = tf.convert_to_tensor(np.array([1]).reshape(-1).astype(np.int32), dtype_hint=tf.int32)
        # valuable_messages = input_dict["obs"]["valuable_messages_indices"]
        for i, transition in enumerate(raw_others_transitions):
            if self._prev_value is not None and self.counter_of_seen_states > 2:  # and self.counter_of_seen_states >= self.adjusting_period:
                h1, c1, *_ = state
                ac_critic_encoded_prev_obs = self.encoder_model(inputs=transition[0])
                ac_pass_dict = {
                    "ac_trunk": ac_critic_encoded_prev_obs
                }
                for k, v in ac_pass_dict.items():
                    ac_pass_dict[k] = add_time_dimension(v, seq_lens)
                _, state = self.forward_rnn(
                    ac_pass_dict, state, seq_lens
                )
                prev_state_values_out = self.value_function()
                # (_, prev_state_values_out, h1, c1,) = self.actions_policy_model.forward_rnn(
                #     ac_pass_dict, [h1, c1], seq_lens
                # )

                ac_critic_encoded_next_obs = self.encoder_model(inputs=transition[-1])
                # Compute the next action
                ac_pass_dict = {
                    "ac_trunk": ac_critic_encoded_next_obs
                }
                for k, v in ac_pass_dict.items():
                    ac_pass_dict[k] = add_time_dimension(v, seq_lens)
                _, state = self.forward_rnn(
                    ac_pass_dict, state, seq_lens
                )
                next_state_value_out = self.value_function()
                # (_, next_state_value_out, h1, c1,) = self.actions_policy_model.forward_rnn(
                #     ac_pass_dict, [h1, c1], seq_lens
                # )
                # 2. calculate td error for this state
                v_current = tf.reduce_max(next_state_value_out)
                r_current = tf.reduce_max(transition[2])
                v_prev = tf.reduce_max(prev_state_values_out)

                # 3. sum abs value of the td-errors
                td_errors_sum += tf.reduce_max(tf.math.abs(tf.cast(r_current, tf.float32) + 0.99 * v_current - v_prev))

        return tf.reduce_max(td_errors_sum), state
