# 测试并行环境 Example_Parallel_Environment

import Example_Custom_Environment as epe

parallel_env = epe.parallel_env(render_mode='human')
observations = parallel_env.reset()

while parallel_env.agents:
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)

