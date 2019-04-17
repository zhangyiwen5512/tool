import torch
import torch.nn.functional as F
import matplotlib as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-5,5,100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x,y= Variable(x),Variable(y)


# method 1
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


    def backward(self):
        pass

    def save(self):
        pass

    def restore_net(self):
        pass

    def restore_para(self):
        pass


opt = torch.optim.SGD()
#hyperparamater





net = Net(1, 10, 1)
print(net)

# medhtd 2  快速搭建网络
net = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)



print(net)


"""
保存和提取网络




"""
