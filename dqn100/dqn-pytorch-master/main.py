import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import scienceplots

import gym
from matplotlib import pyplot as plt

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

from tensorboardX import SummaryWriter

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)  # 需要接近3000000步
    if steps_done % 1000 == 0:
        writer.add_scalar('eps/step', eps_threshold, steps_done)
    steps_done += 1  # 动作步数
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8).bool()
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    # 计算当前状态Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    # print(state.shape) # (84, 84, 4)
    state = state.transpose((2, 0, 1))
    # print(state.shape) # (4, 84, 84)
    state = torch.from_numpy(state)
    # print(state.shape) # torch.Size([4, 84, 84])
    # print(state.unsqueeze(0).shape) # torch.Size([1, 4, 84, 84])
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    print('total_episodes:', n_episodes)
    rewardlist = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)  # steps_done+1
            # print(action)  # 0 1 2 3
            if render:
                env.render()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            # print(reward)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state
            # 经验池满了之后开始学习
            if steps_done > INITIAL_MEMORY:  # 10000
                optimize_model()  # learn

                if steps_done % TARGET_UPDATE == 0:  # 跟新目标网络 1000步
                    target_net.load_state_dict(policy_net.state_dict())

            if done:  # 如果回合结束 则结束回合
                writer.add_scalar('rewards/epi', total_reward, episode)
                break

        if episode % 20 == 0:  # 每20个回合打印一次
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
                writer.close()
        if episode % 200 == 0:  # 每200个回合保存一下模型 基本上1000回合就可明显看出效果了
            torch.save(policy_net, "./model/dqn_pong_model{}.pth".format(episode))
            # torch.save(policy_net.state_dict(), "./model/dqn_pong_model{}.pth".format(episode))
            print('{}_episode_model_save'.format(episode))
        rewardlist.append(total_reward)

    env.close()
    plt.plot(rewardlist)
    plt.xlabel("episode")
    plt.ylabel("episode_reward")
    plt.title('train_reward')
    plt.show()
    return

def test(env, n_episodes, policy, render=True):
    # env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video', force=True)
    episode_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            # action = policy(state.to('cuda')).max(1)[1].view(1,1)
            action = policy(state).max(1)[1].view(1, 1)
            if render:
                env.render()
                # time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                episode_rewards.append(total_reward)
                break

    env.close()

    plt.style.use(['science'])
    plt.figure(figsize=(6,4))
    plt.plot(episode_rewards)
    plt.xlabel("episode")
    plt.ylabel("episode_reward")
    plt.title('test_reward')
    plt.ylim(-10,40)
    plt.show()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)  # 更改环境

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    writer = SummaryWriter(log_dir='scalar')

    # train model
    # train(env, 1300, True)  # 指定你需要训练的回合数1300


    policy_net = torch.load("model/dqn_pong_model1200.pth", map_location= 'cpu')  # 加入想测试的模型
    test(env, 50, policy_net, render=True)

