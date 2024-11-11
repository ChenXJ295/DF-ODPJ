from pettingzoo.atari import boxing_v2
env = space_invaders_v2.env(render_mode="human")
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)
env.close()