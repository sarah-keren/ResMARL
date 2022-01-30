from social_dilemmas.envs.Taxis.multitaxienv.taxi_environment import TaxiEnv
from social_dilemmas.envs.cleanup import CleanupEnv, CleanupEnvWithMessagesSelf, CleanupEnvWithMessagesGlobal, \
    CleanupEnvWithMessagesMandatory
from social_dilemmas.envs.envs_with_perturbations.cleanup_with_perts import CleanupPerturbationsEnv, \
    CleanupPerturbationsEnvWithMessagesSelf, CleanupPerturbationsEnvWithMessagesGlobal, \
    CleanupPerturbationsEnvWithMessagesMandatory
from social_dilemmas.envs.envs_with_perturbations.harvest_with_perts import HarvestPerturbationEnv, \
    HarvestPerturbationsEnvWithMessagesSelf, HarvestPerturbationsEnvWithMessagesGlobal, \
    HarvestPerturbationsEnvWithMessagesMandatory
from social_dilemmas.envs.harvest import HarvestEnv, HarvestEnvWithMessagesSelf, HarvestEnvWithMessagesGlobal, \
    HarvestEnvWithMessagesMandatory
from social_dilemmas.envs.switch import SwitchEnv


def get_env_creator(env, num_agents, args):
    # TODO add here envs with messages and confusion classes
    env_name_to_class = {
        "harvest": HarvestEnv,
        "harvest_msg_self": HarvestEnvWithMessagesSelf,
        "harvest_msg_global": HarvestEnvWithMessagesGlobal,
        "harvest_mandatory": HarvestEnvWithMessagesMandatory,

        "cleanup": CleanupEnv,
        "cleanup_msg_self": CleanupEnvWithMessagesSelf,
        "cleanup_msg_global": CleanupEnvWithMessagesGlobal,
        "cleanup_mandatory": CleanupEnvWithMessagesMandatory,
    }
    if env in list(env_name_to_class.keys()):
        def env_creator(_):
            return env_name_to_class[env](
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )
    else:
        perturbation_magnitude = 70
        if '150' in env:
            perturbation_magnitude = 150
        elif '200' in env:
            perturbation_magnitude = 190
        if 'cleanup' in env:
            env_class = CleanupPerturbationsEnv
            if 'msg_self' in env:
                env_class = CleanupPerturbationsEnvWithMessagesSelf
            elif 'msg_global' in env:
                env_class = CleanupPerturbationsEnvWithMessagesGlobal
            elif 'mandatory' in env:
                env_class = CleanupPerturbationsEnvWithMessagesMandatory
        elif 'harvest' in env:
            env_class = HarvestPerturbationEnv
            if 'msg_self' in env:
                env_class = HarvestPerturbationsEnvWithMessagesSelf
            elif 'msg_global' in env:
                env_class = HarvestPerturbationsEnvWithMessagesGlobal
            elif 'mandatory' in env:
                env_class = HarvestPerturbationsEnvWithMessagesMandatory
        elif 'taxis' in env:
            env_class = TaxiEnv

        def env_creator(_):
            return env_class(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=perturbation_magnitude
            )

    return env_creator
