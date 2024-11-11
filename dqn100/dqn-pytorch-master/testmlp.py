import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import os
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import matplotlib.pyplot as plt
plt.switch_backend('agg')

agent1 = pd.read_csv("D:\强化学习论文\code\dqn100\collect_ram_11.csv")
# agent1lab = pd.read_csv("D:\强化学习论文\code\dqn100\collect_data_ram_label.csv")
agent2 = pd.read_csv("D:\强化学习论文\code\dqn100\collect_ram_22.csv")
# agent2lab = pd.read_csv("D:\强化学习论文\code\dqn100\collect_data_ram_label2.csv")
x1 = agent1.iloc[:, 1:131].values
x2 = agent2.iloc[:, 1:131].values
y1 = agent1.iloc[:, 132].values
y2 = agent2.iloc[:, 132].values
y2[y2==2] = 0
X = np.concatenate((x1, x2))
Y = np.concatenate((y1, y2))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
sc.fit_transform(X_train)
sc.transform(X_test)


# 构建全连接神经网络
class myMLP(nn.Module):
    def __init__(self):
        super(myMLP, self).__init__()

        # 隐藏层1
        self.hidden1 = nn.Linear(in_features=130,  # 第一个隐藏层输入为数据的特征数
                      out_features=60,  # 输出为神经元的个数
                      bias= False  # 默认会有偏置
                      )
        self.act1 = nn.ReLU()

        # 隐藏层2
        self.hidden2 =  nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        # 隐藏层3
        self.hidden3 = nn.Linear(30, 10)
        self.act3 = nn.ReLU()

        # 分类层
        self.hidden4 = nn.Linear(10, 2)  # 二分类
        self.act4 = nn.Sigmoid()


    # 定义前向传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        return x



testnet = myMLP().cuda()
# summary(testnet, input_size=(1, 131))
# 数据转为张量
X_train_nots = torch.tensor(X_train, dtype=torch.float).cuda()
y_train_t = torch.tensor(y_train, dtype= torch.long).cuda()
print(torch.unique(y_train_t))
X_test_nots = torch.tensor(X_test, dtype=torch.float).cuda()
y_test_t = torch.tensor(y_test, dtype=torch.long).cuda()
# y_test_t = torch.nn.functional.one_hot(y_test_t,num_classes=2)
# 用TensorDataset捆绑X和y
train_data_nots = Data.TensorDataset(X_train_nots, y_train_t)
test_data = Data.TensorDataset(X_test_nots, y_test_t)
# 定义数据加载器
train_nots_loader = Data.DataLoader(
    dataset = train_data_nots, # 使用的数据集
    batch_size = 64, # 批量处理样本大小
    shuffle = True# 随机打乱

)
test_nots_loader = Data.DataLoader(
    dataset = test_data, # 使用的数据集
    batch_size = 256, # 批量处理样本大小
)
# 定义优化器
optimizer = Adam(testnet.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss().cuda()
def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)
for epoch in range(1000):
    for step, (b_x, b_y) in enumerate(train_nots_loader):
        # 计算每个batch的损失

        testnet(b_x)
        output = testnet(b_x) # MLP在训练 batch上的输出
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()  # 梯度初始化为0
        train_loss.backward() # 反向传播
        print("epoch: {}, batch: {}, loss: {}".format(epoch, 64, train_loss.data))
        optimizer.step() # 使用梯度进行优化
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_nots_loader):
    outputs = testnet(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))


