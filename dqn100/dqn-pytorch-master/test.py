import gym
import torch
from main import test
import math
import random
import numpy as np
import time

import gym
from matplotlib import pyplot as plt

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
import torch
print(torch.__version__)
print(torch.cuda.is_available())
env = gym.make("PongNoFrameskip-v4")
env = make_env(env)  # 更改环境
policy_net = torch.load("model/dqn_pong_model1000.pth")  # 加入想测试的模型
test(env, 100, policy_net, render=True)