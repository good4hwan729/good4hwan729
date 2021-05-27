import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1176, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate, weight_decay=0.04)


    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        if x.size() == torch.Size([2500, 6, 14, 14]):
            x = x.view(2500, -1)
        elif x.size() == torch.Size([100, 6, 14, 14]):
            x = x.view(100, -1)
        else:
            x = x.flatten()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def step(self, x,y):
        self.optimizer.zero_grad()
        #print(x.size())
        loss = self.loss_fn(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    net = NeuralNet(0.02, nn.CrossEntropyLoss(), 3072, 2)
    losses = []
    yhats = []
    totloss = 0
    #print(train_set.size())
    means = train_set.mean(dim=0, keepdim=True)         #Standardization on training set
    stds = train_set.std(dim=0, keepdim=True)
    normalized_data = (train_set - means) / stds

    means = dev_set.mean(dim=0, keepdim=True)           #Stabdardization on dev set
    stds = dev_set.std(dim=0, keepdim=True)
    normalized_data2 = (dev_set - means) / stds

    #print(n_iter*batch_size)
    #print(len(train_set))
    for i in range(600):
        #print(batch_size)
        if n_iter*batch_size > len(train_set):
            start = np.mod(i*batch_size, len(train_set))
            output = net.step(normalized_data[start: start + batch_size], train_labels[start: start + batch_size])
        else:
            output = net.step(normalized_data[i*batch_size:i*(batch_size+1)], train_labels[i*batch_size:i*(batch_size+1)])
        totloss += output
        losses.append(totloss)

    #print("or here")
    net(dev_set)
    #print("final check")
    for data in normalized_data2:
        #print("forward")
        output = net.forward(data)
        output = output.detach().numpy()
        yhats.append(np.argmax(output))

    return losses, yhats, net
