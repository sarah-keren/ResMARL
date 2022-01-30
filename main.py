from social_dilemmas.envs.envs_with_perturbations.harvest_with_perts import HarvestPerturbationsEnvWithMessagesGlobal

harvest = HarvestPerturbationsEnvWithMessagesGlobal()
harvest.reset()
harvest.render()
harvest.time_step_in_instance = harvest.perturbations_frequency - 1
harvest.step({'agent-0': [3, 2, [-1]]})
harvest.render()
