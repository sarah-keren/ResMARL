import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from config.constants import REWARD_UPPER_BOUND

tf = try_import_tf()


class GlobalNextRewardPredictionsLSTM(RecurrentTFModelV2):
    def __init__(
        self, obs_space, messages_space, num_outputs, num_other_agents, model_config, name, cell_size=64,
    ):
        """
        The LSTM in the Model of Other Agents head.
        :param obs_space: The size of the previous fully-connected layer.
        :param messages_space: vector of messages length
        :param num_outputs: The number of outputs. Normally num_other_agents * action_space.
        :param model_config: The model config dict.
        :param name: The model name.
        :param cell_size: The amount of LSTM units.
        """
        super(GlobalNextRewardPredictionsLSTM, self).__init__(obs_space, messages_space, num_outputs, model_config, name)

        self.cell_size = cell_size
        self.messages_space = messages_space

        # Define input layers
        obs_input_layer = tf.keras.layers.Input(shape=(None, obs_space), name="obs_inputs")

        messages_layer = tf.keras.layers.Input(
            shape=(None, messages_space), name="messages_input"
        )
        concat_input = tf.keras.layers.concatenate([obs_input_layer, messages_layer])

        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=concat_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Postprocess LSTM output with another hidden layer and compute values
        next_reward_predicted = tf.keras.layers.Dense(
            num_other_agents, activation=tf.keras.activations.linear, name=name
        )(lstm_out)

        inputs = [obs_input_layer, seq_in, state_in_h, state_in_c]
        inputs.insert(1, messages_layer)
        outputs = [next_reward_predicted, state_h, state_c]
        self.rnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NextRewardPredictionModel")

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the MOA LSTM.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The MOA predictions and new state.
        """
        rnn_input = [input_dict["curr_obs"], seq_lens] + state
        rnn_input.insert(1, input_dict["one_hot_total_messages"])
        model_out, h, c = self.rnn_model(rnn_input)
        model_out = tf.clip_by_value(model_out, clip_value_min=-REWARD_UPPER_BOUND, clip_value_max=REWARD_UPPER_BOUND)
        return model_out, h, c

    @override(ModelV2)
    def get_initial_state(self):
        """
        The state is supposed to be used for LSTMs, but is abused somewhat to transfer over
        calculations from previous model evaluations.
        :return: Initial state of this model.
        [0] and [1] are LSTM state.
        [2] is action logits.
        [3] is the FC output feeding into the LSTM.
        """
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            # np.zeros(self.messages_space, np.float32),  # reward is a scalar measurement
            np.zeros([self.obs_space], np.float32),
        ]
