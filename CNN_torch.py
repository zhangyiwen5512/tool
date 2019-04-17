import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import matplotlib as plt

#hyper paramaters
EPOCH = 12
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = True


# 得到dataset
train_data = torchvision.datasets.MNIST(
    root='./MNIST_data/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root='./MNINST_data/', train=False
)


#plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
#plt.title('%i'% train_data.train_labels[0])
#plt.show()


# 用loader从dataset中提出data
#torch_dataset = Data.TensorDataset(data_tensor=train_data.train_data, target_tensor=train_data.train_labels)
train_loader = Data.DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=2)

#test_loader = Data.DataLoader(test_data, BATCH_SIZE, shuffle=True, num_workers=2)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.0
test_y = test_data.test_labels[0:2000]




class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32*7*7, 10)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out, x


nets = CNN()# 生成网络
opts = torch.optim.SGD(nets.parameters(), lr=LR)# 定义优化器
loss_funcs = torch.nn.SmoothL1Loss()# 定义lossfucntion



# 训练
for epoch in range(EPOCH):
    for step, batch_X in enumerate(train_loader):
        batch_X = Variable(batch_X)

        output = nets(batch_X)# 前向传播
        loss = loss_funcs(output, test_x)# 计算loss
        opts.zero_grad()# 梯度置0
        loss.backward()# 反向传播
        opts.step()# 修正参数
