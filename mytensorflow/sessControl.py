#session使用

import tensorflow as tf

#一行两列
matrix1 = tf.constant([[3,3]])
#两行一列
matrix2 = tf.constant([[2],
                     [2]])

#矩阵相乘
product = tf.matmul(matrix1,matrix2)
#np.dot(1,2)

'''
#定义
sess = tf.Session()

#执行,接收结果,一个run执行一次
result = sess.run(product)
print(result)
sess.close()
'''

#可以自动关闭
with tf.Session() as sess:
    result = sess.run(product)
    print(result)


