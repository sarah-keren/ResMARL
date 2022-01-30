import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from algorithms.common_funcs_baseline import BaselineResetConfigMixin

tf = try_import_tf()

OTHERS_PREV_ACTIONS = "others_actions"
AGENT_PREV_ACTIONS = "agent_prev_actions"
PREDICTED_ACTIONS = "predicted_actions"
VISIBILITY = "others_visibility"
VISIBILITY_MATRIX = "visibility_matrix"
MESSAGES_REWARD = "messages_intrinsic_reward"
EXTRINSIC_REWARD = "extrinsic_reward"
OTHER_PREDICTED_NEXT_REWARD = "others_predicted_next_reward"
AGENT_PREDICTED_NEXT_REWARD = "agent_predicted_next_reward"
AGENT_PREDICTED_GLOBAL_REWARD = "agent_predicted_global_reward"

# Frozen logits of the policy that computed the action
ACTION_LOGITS = "action_logits"
COUNTERFACTUAL_PREDICTED_REWARDS = "counterfactual_predicted_rewards"
POLICY_SCOPE = "func"


class MessagesScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_messages_reward_weight = config["influence_reward_weight"]
        if any(
                config[key] is None
                for key in ["influence_reward_schedule_steps", "influence_reward_schedule_weights"]
        ):
            self.compute_messages_reward_weight = lambda: self.baseline_messages_reward_weight
        self.messages_reward_schedule_steps = config["influence_reward_schedule_steps"]
        self.messages_reward_schedule_weights = config["influence_reward_schedule_weights"]
        self.timestep = 0
        self.cur_messages_reward_weight = np.float32(self.compute_messages_reward_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_messages_reward_weight_tensor = tf.get_variable(
            "cur_messages_reward_weight",
            initializer=self.cur_messages_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(MessagesScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_messages_reward_weight = self.compute_messages_reward_weight()
        self.cur_messages_reward_weight_tensor.load(
            self.cur_messages_reward_weight, session=self._sess
        )

    def compute_messages_reward_weight(self):
        """ Computes multiplier for influence reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.messages_reward_schedule_steps,
            self.messages_reward_schedule_weights,
        )
        return weight * self.baseline_messages_reward_weight


class GlobalRewardPredictorLoss(object):
    def __init__(self, global_reward_preds, true_global_rewards_preds, loss_weight=1.0, others_visibility=None,
                 message="NextReward L2 loss"):
        """Train NextRewardPredictor model with supervised cross entropy loss on a trajectory.
        The model is trying to predict self next reward actions at timestep t+1 given all
        obs at timestep t.
        Returns:
            A scalar loss tensor (l2 loss).
        """
        self.total_loss = tf.reduce_mean(tf.norm(global_reward_preds - true_global_rewards_preds)) * loss_weight
        tf.Print(self.total_loss, [self.total_loss], message=message)


class SelfRewardPredictorLoss(object):
    def __init__(self, self_reward_preds, true_rewards_recieved, loss_weight=1.0, others_visibility=None,
                 message="NextReward L2 loss"):
        """Train NextRewardPredictor model with supervised cross entropy loss on a trajectory.
        The model is trying to predict self next reward actions at timestep t+1 given all
        obs at timestep t.
        Returns:
            A scalar loss tensor (l2 loss).
        """
        self.total_loss = tf.reduce_mean(tf.norm(self_reward_preds - true_rewards_recieved)) * loss_weight
        tf.Print(self.total_loss, [self.total_loss], message=message)


def setup_global_next_reward_prediction_loss(logits, policy, train_batch):
    # Instantiate the prediction loss
    # actual_actions_taken = train_batch[AGENT_PREV_ACTIONS]
    global_next_reward_preds = train_batch[AGENT_PREDICTED_GLOBAL_REWARD]
    # global_next_reward_preds = tf.reshape(global_next_reward_preds, [-1])
    true_global_predicted_rewards = tf.cast(train_batch[OTHER_PREDICTED_NEXT_REWARD], tf.float32)
    # 0/1 multiplier array representing whether each agent is visible to
    # the current agent.
    if policy.train_messages_only_when_visible:
        # if VISIBILITY in train_batch:
        others_visibility = train_batch[VISIBILITY]
    else:
        others_visibility = None
    global_next_reward_loss = GlobalRewardPredictorLoss(
        global_next_reward_preds,
        true_global_predicted_rewards,
        loss_weight=policy.messages_loss_weight,
        others_visibility=others_visibility,
    )
    return global_next_reward_loss


def setup_next_reward_prediction_loss(logits, policy, train_batch):
    # Instantiate the prediction loss
    # actual_actions_taken = train_batch[AGENT_PREV_ACTIONS]
    next_reward_preds = train_batch[AGENT_PREDICTED_NEXT_REWARD][:, 0, 0]
    next_reward_preds = tf.reshape(next_reward_preds, [-1])
    true_rewards = train_batch[EXTRINSIC_REWARD]
    # 0/1 multiplier array representing whether each agent is visible to
    # the current agent.
    if policy.train_messages_only_when_visible:
        # if VISIBILITY in train_batch:
        others_visibility = train_batch[VISIBILITY]
    else:
        others_visibility = None
    next_reward_loss = SelfRewardPredictorLoss(
        next_reward_preds,
        true_rewards,
        loss_weight=policy.messages_loss_weight,
        others_visibility=others_visibility,
    )
    return next_reward_loss


def msg_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Weigh social influence reward and add to batch.
    sample_batch = weigh_and_add_msg_reward(policy, sample_batch)

    return sample_batch


def weigh_and_add_msg_reward(policy, sample_batch):
    cur_messages_reward_weight = policy.compute_messages_reward_weight()
    # Since the reward calculation is delayed by 1 step, sample_batch[SOCIAL_INFLUENCE_REWARD][0]
    # contains the reward for timestep -1, which does not exist. Hence we shift the array.
    # Then, pad with a 0-value at the end to make the influence rewards align with sample_batch.
    # This leaks some information about the episode end though.
    messages_intrinsic_rewards = sample_batch[MESSAGES_REWARD]

    # Clip and weigh influence reward
    messages_intrinsic_rewards = np.clip(messages_intrinsic_rewards, -policy.messages_reward_clip,
                                         policy.messages_reward_clip)
    messages_intrinsic_rewards = messages_intrinsic_rewards * cur_messages_reward_weight

    # Add to trajectory
    sample_batch[MESSAGES_REWARD] = messages_intrinsic_rewards
    sample_batch["extrinsic_reward"] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + messages_intrinsic_rewards

    return sample_batch


def agent_name_to_idx(agent_num, self_id):
    """split agent id around the index and return its appropriate position in terms
    of the other agents"""
    agent_num = int(agent_num)
    if agent_num > self_id:
        return agent_num - 1
    else:
        return agent_num


def get_agent_visibility_multiplier(trajectory, num_other_agents, agent_ids):
    traj_len = len(trajectory["obs"])
    visibility = np.zeros((traj_len, num_other_agents))
    for i, v in enumerate(trajectory[VISIBILITY]):
        vis_agents = [agent_name_to_idx(a, agent_ids[i]) for a in v]
        visibility[i, vis_agents] = 1
    return visibility


def extract_last_actions_from_episodes(episodes, batch_type=False, own_actions=None):
    """Pulls every other agent's previous actions out of structured data.
    Args:
        episodes: the structured data type. Typically a dict of episode
            objects.
        batch_type: if True, the structured data is a dict of tuples,
            where the second tuple element is the relevant dict containing
            previous actions.
        own_actions: an array of the agents own actions. If provided, will
            be the first column of the created action matrix.
    Returns: a real valued array of size [batch, num_other_agents] (meaning
        each agents' actions goes down one column, each row is a timestep)
    """
    if episodes is None:
        print("Why are there no episodes?")
        import ipdb

        ipdb.set_trace()

    # Need to sort agent IDs so same agent is consistently in
    # same part of input space.
    agent_ids = sorted(episodes.keys())
    prev_actions = []

    for agent_id in agent_ids:
        if batch_type:
            prev_actions.append(episodes[agent_id][1]["actions"])
        else:
            prev_actions.append([e.prev_action for e in episodes[agent_id]])

    all_actions = np.transpose(np.array(prev_actions))

    # Attach agents own actions as column 1
    if own_actions is not None:
        all_actions = np.hstack((own_actions, all_actions))

    return all_actions


def messages_fetches(policy):
    """Adds logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        # Be aware that this is frozen here so that we don't
        # propagate agent actions through the reward
        ACTION_LOGITS: policy.model.action_logits(),
        OTHERS_PREV_ACTIONS: policy.model.other_agent_actions(),
        AGENT_PREV_ACTIONS: policy.model.agent_prev_actions(),
        VISIBILITY: policy.model.visibility(),
        MESSAGES_REWARD: policy.model.messages_intrinsic_reward(),
        AGENT_PREDICTED_NEXT_REWARD: policy.model.predicted_next_reward(),
        AGENT_PREDICTED_GLOBAL_REWARD: policy.model.global_next_reward_prediction(),
        OTHER_PREDICTED_NEXT_REWARD: policy.model.global_other_predicted_reward()
    }


class MessagesConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.num_other_agents = config["num_other_agents"]
        self.messages_loss_weight = tf.get_variable(
            "messages_loss_weight", initializer=config["moa_loss_weight"], trainable=False
        )
        self.messages_reward_clip = config["influence_reward_clip"]
        self.train_messages_only_when_visible = False
        self.message_only_when_visible = config["influence_only_when_visible"]


class MessagesResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config, session):
        custom_options = new_config["model"]["custom_options"]
        for policy in policies:
            policy.messages_loss_weight.load(custom_options["moa_loss_weight"], session=session)
            policy.compute_messages_reward_weight = lambda: custom_options[
                "influence_reward_weight"
            ]

    def reset_config(self, new_config):
        policies = self.optimizer.policies.values()
        BaselineResetConfigMixin.reset_policies(policies, new_config)
        self.reset_policies(policies, new_config, self.optimizer.sess)
        self.config = new_config
        return True


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space, action_space, logit_dim, config["model"], name=POLICY_SCOPE, framework="tf",
    )

    return policy.model


def setup_messages_mixins(policy, obs_space, action_space, config):
    MessagesScheduleMixIn.__init__(policy, config)
    MessagesConfigInitializerMixIn.__init__(policy, config)


def get_messages_mixins():
    return [
        MessagesConfigInitializerMixIn,
        MessagesScheduleMixIn,
    ]


def validate_messages_config(config):
    config = config["model"]["custom_options"]
    if config["influence_reward_weight"] < 0:
        raise ValueError("Influence reward weight must be >= 0.")
