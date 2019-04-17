#coding:utf-8
#反向传播算法
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#返回标签。从x里取出一行判断
def judge(x0,x1):		
				
		t = x0 + x1	
		if(t < 1) :
			return 1
		else :
			return 0


#产生随机数
rng = np.random.RandomState(seed)
#初始化输入和标签，生成32行2列
X = rng.rand(32,2)
Y = [[judge(x0,x1)] for (x0,x1) in X]
#print "X:\n",X
#print "Y:\n",Y


#定义输入，参数，输出，传播
#32组2列
x_data = tf.placeholder(tf.float32, shape=(None,2))
#32组1列
y_data = tf.placeholder(tf.float32, shape=(None,1))

#定义W
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播
a = tf.matmul(x_data,w1)
y = tf.matmul(a,w2)


#定义loss和BP
loss = tf.reduce_mean(tf.square(y-y_data))
#三种反向传播选其一,learning_rate=0.001
#梯度下降
print ("GradientDescent")
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
'''
#动量方法
print ("Momentum")
train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
'''
#Adam算法
#print ("Adam")
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


#初始化参数并训练
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
#未经训练的W
	ws1 = sess.run(w1)
	ws2 = sess.run(w2)
#训练模型
	STEP = 3000
	for i in range(STEP):
#计算batch的初始和结束
		start = (i*BATCH_SIZE)%32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x_data:X[start:end],y_data:Y[start:end]})
		if	i %500 == 0 :
			total_loss =sess.run(loss, feed_dict={x_data:X,y_data:Y})
			print("After %d training,loss = %g" %(i,total_loss))

#训练后的W
	wd1=sess.run(w1)
	wd2=sess.run(w2)



#输出结果
print ("ws1\n%s" %ws1)
print ("ws2\n%s" %ws2)
print ("result\n%s" %result)
print ("wd1\n%s" %wd1)
print ("wd2\n%s" %wd2)




















