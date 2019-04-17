#coding:utf-8
import tensorflow as tf
a= tf.constant([[1.0],[2.0]])
b= tf.constant([[1.0,2.0]])
#a.shape()
result1=tf.matmul(a,b)
result2=tf.matmul(b,a)

with tf.Session() as sess:
	print (sess.run(result1))
	print (sess.run(result2))

print a
print b
print result1
print result2

#Tensor("add:0",shape=(2,),dtype=float32)
