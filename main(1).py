from pettingzoo.classic import texas_holdem_v4
env = texas_holdem_v4.env()

env.reset()

for agent in env.agent_iter():
    env.render()
    observation, reward, done, info = env.last()
    print('state = {0}; reward = {1}'.format(observation, reward))
    if done :
        action = None
    else:
        action = env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)
env.close()