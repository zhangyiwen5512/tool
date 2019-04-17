import torch
import numpy as np
from torch.autograd import Variable# 用Variable变量存储参数
import torch.nn.functional as F # 激励函数
import matplotlib.pyplot as plt
import os
import glob





snapshot_file = os.path.join("/home/zhangyiwen/Desktop/tensorflow/" + "*.pkl")
snapshot_file = glob.glob(snapshot_file)
snapshot_file.sort(key=os.path.getmtime)
print(snapshot_file)
file_number = len(snapshot_file)
if file_number > 4:
    snapshot_file = snapshot_file[0:file_number - 4]
    print(snapshot_file)

print("delete model to {}".format(snapshot_file))

for c in snapshot_file:
    os.remove(c)
    #np_paths.remove(nfile)





exit(0)





# 数据制作
x = torch.linspace(-5,5,200)
x = Variable(x, requires_grad=True)
xnp= x.data.numpy()

yrelu = F.relu(x).data.numpy()
ysigmoid = F.sigmoid(x).data.numpy()
ytanh = F.tanh(x).data.numpy()
#ysoftmax = F.softmax(x).data.numpy()

plt.figure(num=1,figsize=(8,6))
plt.ylim(-1,5)
plt.plot(xnp,yrelu,c='red',label='relu')

plt.figure(num=2,figsize=(8,6))
plt.ylim(-1.2,1.2)
plt.plot(xnp,ysigmoid,c='blue',label='sigmoid')

plt.figure(num=3,figsize=(8,6))
plt.ylim(-2,2)
plt.plot(xnp,ytanh,c='green',label='tanh')

plt.show()





"""







tensor = torch.Tensor([[1,2],[12,11]])
var = Variable(tensor, requires_grad=True)# 搭建计算图纸， requires_gard 计算梯度

out = torch.mean(tensor * tensor)
out2 = torch.mean(var * var)

out2.backward()
print(var.data,var.grad)








#[2,3]
np_data = np.arange(6).reshape((2,3))
# np转行torch
torch_data = torch.from_numpy(np_data)
# torch转行np
tensor = torch_data.numpy()

print(np_data,'\n',torch_data,'\n',tensor)


#data = [1,2,1,2]
#tensor = torch.FloatTensor(data) #32位

data = [[1,2],[3.4]]
data = np.array(data,dtype=float)
tensor = torch.from_numpy(data)

print(
    '\nnumpy:',np.multiply(data,data),data.dot(data),# 矩阵相乘
    '\ntorch:', torch.mm(tensor,tensor),torch.dot(data)# torch，dot是矩阵展平后相乘的结果
)




"""


