from gym.envs.registration import register

register(
    id='simple-env-clean-v0',
    entry_point='gym_adlr.envs:SimpleEnvClean',
)
