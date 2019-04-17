import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#下载数据
mnist = input_data.read_data_sets("/home/zhangyiwen/桌面/tensorflow/MNIST_data/",one_hot=True)





#(输入数据，输入维度，输出维度，激励函数)
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    ###         layer块             #####
    layer_name = 'LAYER%s' % n_layer #层名命名
    with tf.name_scope(layer_name):
        with tf.name_scope('WEIGHTS'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram('/weights',Weights)
        with tf.name_scope('BIASES'):
            biases = tf.Variable(tf.zeros([out_size]) + 0.1,name='B')
            tf.summary.histogram('/biases',biases)
        #y'=w*x+b
        with tf.name_scope('WX_PLUS_B'):
            Wx_plus_b = tf.nn.softmax(tf.matmul(inputs,Weights) + biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            #激励函数处理  
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('/outputs',outputs)  
    return outputs


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result



#定义placeholder
#with tf.name_scope('INPUT'):
xs = tf.placeholder(tf.float32,[None,28*28],name='xInPut')
ys = tf.placeholder(tf.float32,[None,10],name='yInPut')

#加入output层
prediction = add_layer(xs,784,10,1,activation_function=tf.nn.softmax)

#loss function
with tf.name_scope('CROSS_ENTROPY'):    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                reduction_indices=[1],name='cross_entropy'),name='mean')                      
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('TRAIN'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()


#train

with tf.Session() as sess:
     #加载文件到浏览器,保存路径
    ###         浏览器查看          ###
    merged = tf.summary.merge_all()#所有summary打包
    ###切换到目录，tensroboard --logdir='目录'
    writer = tf.summary.FileWriter("/home/zhangyiwen/桌面/tensorflow/mytensorflow/graph",sess.graph)
    sess.run(init)
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)#SGD
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i % 50 == 0:
            ###         run merge
            result = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys})
            #summery写入writer，步数
            writer.add_summary(result,i)
            print(compute_accuracy(mnist.test.images,mnist.test.labels))