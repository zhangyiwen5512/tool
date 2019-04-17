import tensorflow as tf 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#加载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)




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
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)#dropout功能
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



#dropout
keep_prob = tf.placeholder(tf.float32)

#定义placeholder
with tf.name_scope('INPUT'):
    xs = tf.placeholder(tf.float32,[None,8*8],name='xInPut')
    ys = tf.placeholder(tf.float32,[None,10],name='yInPut')

#加入output层
layer1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(layer1,50,10,'l2',activation_function=tf.nn.softmax)

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
    train_writer = tf.summary.FileWriter("/home/zhangyiwen/桌面/tensorflow/mytensorflow/graph/train",sess.graph)
    test_writer = tf.summary.FileWriter("/home/zhangyiwen/桌面/tensorflow/mytensorflow/graph/test",sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.6})
        if i % 50 == 0:
            ###         run merge
            train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
            test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
            #summery写入writer，步数
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)
            #print(compute_accuracy(mnist.test.images,mnist.test.labels))