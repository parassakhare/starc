from gymnasium.envs.registration import register

register(
    id="starc/CubeTransfer-v0",
    entry_point="sim.starc_env:StarcCubeTransferEnv",
    max_episode_steps=500,
)
