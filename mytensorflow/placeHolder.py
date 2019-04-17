import tensorflow as tf 

#placeHoulder,站位符,计算过程中输入数据
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

#对应元素相乘
output = tf.multiply(input1,input2)

#
with tf.Session() as sess:
    #通过feed_dict={}字典给对应数据赋值
    print(sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))
    