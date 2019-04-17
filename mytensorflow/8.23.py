import tensorflow as tf
import numpy as np 

#create data
x_data= np.random.rand(100).astype(np.float32)
y_data= x_data * 0.1 + 0.3

### create structure start   ###

#w,初始-1到1
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
#b,初始为0
biases = tf.Variable(tf.zeros([1]))
#拟合函数
y = Weight * x_data + biases
#损伤函数
loss = tf.reduce_mean(tf.square(y - y_data))

#优化器,减少误差，学习率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化
init = tf.global_variables_initializer()

### create structure end     ###

###     训练                  ###
sess = tf.Session()
sess.run(init)#激活

for setp in range(201):
    sess.run(train)         #训练
    if setp %   20  ==  0:
        print(setp,sess.run(Weight),sess.run(biases))

###     结束                  ###

