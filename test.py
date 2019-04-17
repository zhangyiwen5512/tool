#coding:utf-8
#age=input("input age\n")
#if age>18 :
#	print "true"
#else:
#	print "false"

#t=turtle.Pen()
#for i in range (0,24):
#	t.forward(100)
#	t.right(60)
#`turtle.mainloop()
#python 生成三维数据用一个平面拟合
import tensorflow as tf
import numpy as np

#生成100个假数据，随机
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.100,0.2], x_data)+0.300

#构造一个线性模型
b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(w,x_data)+b

#最小化方差
loss= tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#启动图
sess = tf.Session()
sess.run(init)

#拟合平面
for step in range (0,201):
	sess.run(train)
	if step % 20 == 0:
		print step, sess.run(w), sess.run(b)



sess.close()
#最佳结果w:[[0.1,0.2]],b:[0.3]

