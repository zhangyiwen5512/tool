#coding:utf-8
#两层神经网络
import  tensorflow as tf

#定义输入和参数
#x_data = tf.constant([[0.7,0.5]])

#placehold 喂入多组数
#一组两个特征参数
x_data = tf.placeholder(tf.float32,shape=(1,2))
#多组两个特征参数
x_data = tf.placeholder(tf.float32,shape=(None,2))

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播
a = tf.matmul(x_data,w1)
y = tf.matmul(a,w2)

#初始化并计算
with tf.Session() as sess:
#	init_op = tf.initialize_all_variables()    #不建议使用
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
#	result = sess.run(y)
	
	
#喂入数据2D
#	result = sess.run(y,feed_dict={x_data:[[0.7,0.5]]})
	result = sess.run(y,feed_dict={x_data:
	[[0.7,0.5],
	 [0.2,0.3],
	 [0.3,0.4],
	 [0.4,0.5]]})
	ww1=sess.run(w1)
	ww2=sess.run(w2)
	

#输出结果
print ("result\n%s" %result)
print ("w1\n%s" %ww1)
print ("w2\n%s" %ww2)
