from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from config.constants import REWARD_UPPER_BOUND, EPSILON_CONSTANT
from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.global_next_reward_lstm import GlobalNextRewardPredictionsLSTM
from models.next_reward_lstm import NextRewardLSTM

tf = try_import_tf()


class MessagesWithGlobalConfusionModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        A base model that uses messages to reduce its self confusion.

        :param obs_space: The observation space shape.
        :param action_space: The amount of available actions to this agent.
        :param num_outputs: The amount of available actions to this agent.
        :param model_config: The model config dict. Used to determine size of conv and fc layers.
        :param name: The model name.
        """
        super(MessagesWithGlobalConfusionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self._other_agent_actions = None
        self._visibility = None
        self._intrinsic_reward = None
        self._counterfactuals_general_reward_predictions = None
        self._self_next_reward_pred = None
        self._global_next_reward_prediction_pred = None
        self._global_other_predicted_reward = None

        self._actions_model_out = None
        self._actions_value_out = None
        self._value_out = None
        self._messages_model_out = None
        self._model_out = None
        self._messages_value_out = None
        self._true_one_hot_messages = None
        self._agent_prev_actions = None

        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.actions_num_outputs = int((num_outputs - 1) / 2)
        self.messages_num_outputs = int((num_outputs - 1) / 2)

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

        inner_obs_space_with_messages = inner_obs_space + self.num_other_agents
        inner_obs_space_with_messages_and_predicted_reward = inner_obs_space + self.num_other_agents * 2

        # note that action space is [action, message, conf_level]
        self.actions_policy_model = ActorCriticLSTM(
            inner_obs_space_with_messages,
            action_space[0],
            self.actions_num_outputs,
            model_config,
            "actions_policy",
            cell_size=cell_size,
        )
        self.messages_policy_model = ActorCriticLSTM(
            inner_obs_space_with_messages_and_predicted_reward,
            action_space[1],
            self.messages_num_outputs,
            model_config,
            "messages_policy",
            cell_size=cell_size,
        )
        self.next_reward_prediction_model = NextRewardLSTM(
            inner_obs_space,
            action_space[1].n * (self.num_other_agents + 1),
            self.actions_num_outputs,
            model_config,
            "next_reward_model",
            cell_size=cell_size,
        )

        self.global_next_reward_prediction_prediction_model = GlobalNextRewardPredictionsLSTM(
            obs_space=inner_obs_space,
            messages_space=action_space[1].n * (self.num_other_agents + 1),
            num_outputs=self.actions_num_outputs,
            num_other_agents=self.num_other_agents,
            model_config=model_config,
            name="global_next_reward_model",
            cell_size=cell_size,
        )

        self.register_variables(self.actions_policy_model.rnn_model.variables)
        self.actions_policy_model.rnn_model.summary()

        self.register_variables(self.messages_policy_model.rnn_model.variables)
        self.messages_policy_model.rnn_model.summary()

        self.register_variables(self.next_reward_prediction_model.rnn_model.variables)
        self.next_reward_prediction_model.rnn_model.summary()

        self.register_variables(self.global_next_reward_prediction_prediction_model.rnn_model.variables)
        self.global_next_reward_prediction_prediction_model.rnn_model.summary()

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
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

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
        self._other_agent_actions = input_dict["obs"]["other_agent_actions"]
        self._visibility = input_dict["obs"]["visible_agents"]
        self._agent_prev_actions = input_dict["prev_actions"]
        self._global_other_predicted_reward = input_dict["obs"]["other_agent_predicted_rewards"]

        ac_critic_encoded_obs = self.encoder_model(inputs=input_dict["obs"]["curr_obs"])
        rnn_input_dict = {
            "ac_trunk": ac_critic_encoded_obs,
            "other_agent_actions": input_dict["obs"]["other_agent_actions"],
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": input_dict["prev_actions"],
            "other_agent_messages": input_dict["obs"]["other_agent_messages"],
            "other_agent_predicted_rewards": input_dict["obs"]["other_agent_predicted_rewards"],
            "other_agent_actual_rewards": input_dict["obs"]["other_agent_actual_rewards"]
        }

        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        output, new_state = self.forward_rnn(input_dict=rnn_input_dict,
                                             state=state,
                                             seq_lens=seq_lens)
        # output here should've values for (actions, messages, self_confusion)
        self.compute_intrinsic_reward(input_dict)

        return tf.reshape(output, [-1, self.num_outputs]), new_state  # + [ac_critic_encoded_obs] * 2

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
        h1, c1, h2, c2, h3, c3, tmp1, h4, c4, tmp2 = state

        # Compute the next action
        ac_pass_dict = {
            "curr_obs": tf.concat(
                [input_dict["ac_trunk"], tf.cast(input_dict["other_agent_messages"], dtype=tf.float32)], axis=-1)
        }
        (self._actions_model_out, self._actions_value_out, output_h1,
         output_c1,) = self.actions_policy_model.forward_rnn(
            ac_pass_dict, [h1, c1], seq_lens
        )

        messages_pass_dict = {
            "curr_obs": tf.concat(
                [input_dict["ac_trunk"], tf.cast(input_dict["other_agent_messages"], dtype=tf.float32),
                 tf.cast(input_dict["other_agent_predicted_rewards"], dtype=tf.float32)], axis=-1)
        }

        (self._messages_model_out, self._messages_value_out, output_h2,
         output_c2,) = self.messages_policy_model.forward_rnn(
            messages_pass_dict, [h2, c2], seq_lens
        )

        other_messages = input_dict["other_agent_messages"]
        agent_messages = tf.expand_dims(input_dict["prev_actions"][:, :, 1], axis=-1)
        all_messages = tf.concat([tf.cast(agent_messages, tf.uint8), other_messages], axis=-1,
                                 name="concat_true_messages")
        self._true_one_hot_messages = self._reshaped_as_one_hot(all_messages, self.messages_num_outputs,
                                                                "forward_one_hot_messages")

        reward_predictor_pass_dict = {
            "curr_obs": input_dict["ac_trunk"],
            "one_hot_total_messages": self._true_one_hot_messages,
            "values_predicted": self._actions_model_out
        }

        self._self_next_reward_pred, output_h3, output_c3 = self.next_reward_prediction_model.forward_rnn(
            reward_predictor_pass_dict, [h3, c3], seq_lens
        )
        self._self_next_reward_pred = tf.concat([self._self_next_reward_pred, self._self_next_reward_pred], axis=-1)
        self._self_next_reward_pred = tf.clip_by_value(self._self_next_reward_pred, clip_value_min=-REWARD_UPPER_BOUND,
                                                       clip_value_max=REWARD_UPPER_BOUND)

        general_reward_predictor_pass_dict = {
            "curr_obs": input_dict["ac_trunk"],
            "one_hot_total_messages": self._true_one_hot_messages,
        }

        self._global_next_reward_prediction_pred, output_h4, output_c4 = self.global_next_reward_prediction_prediction_model.forward_rnn(
            general_reward_predictor_pass_dict, [h4, c4], seq_lens
        )

        # computing counterfactual immediate reward assuming different messages
        counterfactuals_general_reward_predictions = []

        for i in range(self.messages_num_outputs):
            counterfactual_messages = tf.pad(
                other_messages, paddings=[[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=i
            )

            one_hot_counterfactual_messages = self._reshaped_as_one_hot(
                counterfactual_messages, self.messages_num_outputs, "messages_with_counterfactual_one_hot"
            )

            general_reward_predictor_pass_dict = {
                "curr_obs": input_dict["ac_trunk"],
                "one_hot_total_messages": one_hot_counterfactual_messages,
            }

            counterfactual_general_next_reward_pred, _, _ = self.global_next_reward_prediction_prediction_model.forward_rnn(
                general_reward_predictor_pass_dict, [h4, c4], seq_lens
            )
            counterfactuals_general_reward_predictions.append(counterfactual_general_next_reward_pred)

        self._counterfactuals_general_reward_predictions = tf.concat(
            counterfactuals_general_reward_predictions, axis=-1, name="concat_counterfactuals_reward_preds"
        )

        self._other_agent_actions = input_dict["other_agent_actions"]
        self._visibility = input_dict["visible_agents"]

        self._model_out = tf.reshape(
            tf.concat([self._actions_model_out, self._messages_model_out, self._self_next_reward_pred], axis=-1),
            [-1, self.num_outputs])
        self._value_out = tf.reshape(self._actions_value_out, [-1])

        return self._model_out, [output_h1, output_c1, output_h2, output_c2, output_h3, output_c3, tmp1, output_h4, output_c4, tmp2]

    def compute_intrinsic_reward(self, input_dict):
        """
        We have the actual reward + we have the estimated reward given our vector of messages.
        So given some predicted R we can calculate the achieved confusion.
        We define the negative of the reward (of the messages) to be the inverse of:
        The dist(l1, l2...) between the minimal confusion and the actual confusion.
        """
        # prev_actions = input_dict["prev_actions"]
        prev_general_rewards = tf.cast(input_dict["obs"]["other_agent_actual_rewards"], tf.float32)
        counterfactual_general_rewards = tf.cast(tf.reshape(
            self._counterfactuals_general_reward_predictions,
            [-1, self.num_other_agents, self.messages_num_outputs]), tf.float32)
        predicted_general_rewards = tf.cast(input_dict["obs"]["other_agent_predicted_rewards"], tf.float32)
        actual_general_rewards = prev_general_rewards
        general_current_confusion_level = tf.math.divide(
            tf.math.abs(predicted_general_rewards - actual_general_rewards),
            tf.math.abs(actual_general_rewards) + EPSILON_CONSTANT,
            name='actual_confusion_levels')

        expanded_actual_rewards_for_counterfactuals = tf.reshape(
            tf.concat([tf.expand_dims(actual_general_rewards, axis=0)] * counterfactual_general_rewards.shape[-1].value,
                      axis=-1),
            [-1, self.num_other_agents, counterfactual_general_rewards.shape[-1].value])
        counterfactual_confusion_levels_preds = tf.math.divide(
            tf.math.abs(tf.math.subtract(counterfactual_general_rewards,
                                         expanded_actual_rewards_for_counterfactuals)),
            tf.math.abs(expanded_actual_rewards_for_counterfactuals) + EPSILON_CONSTANT,
            name=f'hypothetical_confusion_levels')

        min_counterfactual_confusion_levels_preds = tf.math.reduce_min(counterfactual_confusion_levels_preds,
                                                                       axis=-1)
        self._intrinsic_reward = -tf.norm(tf.abs(
            tf.math.subtract(general_current_confusion_level, min_counterfactual_confusion_levels_preds)), axis=-1)

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

    def counterfactual_predicted_rewards(self):
        return self._counterfactuals_general_reward_predictions

    def messages_intrinsic_reward(self):
        return self._intrinsic_reward

    def predicted_next_reward(self):
        return self._self_next_reward_pred

    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        """
        :return: Initial state of this model. This model only has LSTM state from the policy_model.
        """
        return self.actions_policy_model.get_initial_state() + self.messages_policy_model.get_initial_state() + \
               self.next_reward_prediction_model.get_initial_state() + \
               self.global_next_reward_prediction_prediction_model.get_initial_state()

    def agent_prev_actions(self):
        return self._agent_prev_actions

    def global_next_reward_prediction(self):
        return self._global_next_reward_prediction_pred

    def global_other_predicted_reward(self):
        return self._global_other_predicted_reward

    # todo - when writing the mandatory - add here "from_batch" override
