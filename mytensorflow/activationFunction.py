import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt#可视化模块



#(输入数据，输入维度，输出维度，激励函数)
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    #y'=w*x+b
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        #激励函数处理  
        outputs = activation_function(Wx_plus_b)
    return outputs

#生成300个在-1到1的数据
x_data = np.linspace(-1,1,300)[:,np.newaxis] #行转列  
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
 
#None=banchsize
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])



#input layer
#hidden layer 10个神经元
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

#output layer 1个神经元
prediction = add_layer(l1,10,1,activation_function=None)

#损失函数,平方求和取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
            reduction_indices=[1]))#按行求和

#优化器,减少误差，学习率0.5,如何训练
train_step = tf.train.AdamOptimizer().minimize(loss)

#初始化,否则定义的variable不工作
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()#生成图片框
    ax = fig.add_subplot(1,1,1)#加上axis,
    ax.scatter(x_data,y_data)#真实数据加入
    plt.ion()#继续执行主程序
    for i in range(1000):
        sess.run(train_step,feed_dict = {xs:x_data, ys:y_data})
        if i % 50 == 0:
            #total_loss = sess.run(loss,feed_dict={xs:x_data,ys:y_data})
            #print(total_loss)
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.5)
        if  i  == 1000:
            print("end")
    
    
    plt.ioff()#保留图像
    plt.show()#显示