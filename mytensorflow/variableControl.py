import tensorflow as tf 

#定义为变量才变化，否则为常量
state = tf.Variable(0,name='counter')
#print(state.name)
#常量
one = tf.constant(1)

new_value = tf.add(state , one)
#更新
update = tf.assign(state, new_value)

#初始化,否则定义的variable不工作
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)#否则不初始化
    for i in range(3):
        sess.run(update)
        print(sess.run(state))