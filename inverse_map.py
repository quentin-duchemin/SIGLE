    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import os

def sigmoid(u):
    return (1/(1+np.exp(-u)))
def sigmoid1(a):
    return sigmoid(a)*(1-sigmoid(a))
def sigmoid2(a):
    return (1-2*sigmoid(a))*sigmoid1(a)
def logit(a):
    return np.log(a/(1-a))

class Net(nn.Module):

    def __init__(self, s):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s, 120)
        self.fc2 = nn.Linear(120, 300)
        self.fc3 = nn.Linear(300, 120)
        self.fc4 = nn.Linear(120,s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_network(matX,thetalim,nb_epochs=6000, lrstart=0.01, lrdecay_step=1000, lrdecay_factor=0.1):
    n,s = matX.shape

    loss_values = []
    net = Net(s)
    optimizer = optim.SGD(net.parameters(), lr=lrstart)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lrdecay_step, gamma=lrdecay_factor)

    for i in range(nb_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        target = np.random.normal(0,1,(1000, s)) * np.tile(thetalim*np.random.rand(1000).reshape(-1,1),(1,s))
        input = (matX.T @ sigmoid(matX @ target.T)).T
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()
        output = net(input)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        loss_values.append(loss.item())
        optimizer.step()    # Does the update
        scheduler.step()
    return net, loss_values

def inverse_map(n=3,d=2, net=None, matX=None, rho=None, seed=None, display=False):
    assert n>d,'We need to choose n larger than d.'
    # Parameters
    if seed is not None:
        np.random.seed(seed)
    if matX is None:
        matX =  np.random.normal(0,1,(n,d))
    else:
        n, d = np.shape(matX)
    if rho is not None:
        # Observation of the projection of the point of interest (this is observed)
        proj = rho
    else:
        if theta is None:
            theta = np.random.normal(0,3,d)
        # Observation of the projection of the point of interest (this is observed)
        proj = matX.T @ sigmoid(matX @ theta)


    theta0 = net(torch.from_numpy(proj).float())
    theta0 = theta0.detach().numpy()
    C = (proj-matX.T @ sigmoid(matX @ theta0))
    U = theta0

    # Initial speed
    dottheta0 = np.linalg.inv(matX.T @  np.diag(sigmoid1(matX @ theta0)) @ matX ) @ C
    Up = dottheta0

    # ODE
    def ode(y, t):
        res = np.zeros(2*d)
        res[:d] = y[d:]
        res[d:] = - np.dot( np.linalg.inv(matX.T @ np.diag(sigmoid1(matX @ y[:d])) @ matX)@ matX.T, sigmoid2(matX @ y[:d])*((matX @ y[d:])**2) )
        return res
    t = np.linspace(0,1,700)
    y0 = np.hstack((U,Up))
    sol = odeint(ode, y0, t)
    return sol[len(t)-1,:d], sigmoid(matX @ sol[len(t)-1,:d])    
