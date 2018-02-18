# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:39:00 2018

@author: pc
"""
#mnist 데이터 다운로드
import matplotlib.pylab as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
#변수설정
x=tf.placeholder(tf.float32, [None, 784])
w=tf.Variable(tf.zeros(784,10))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
#cross-entropy 모델 설정
y_=tf.placeholder(tf.float32, [None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#경사하강법 모델 학습
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    _, cross_entropy_value = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    print(cross_entropy_value)
    sess.run(train_step, feed_dict={x: batch_xs,y_:batch_ys})
    
#모델 정확도
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))    
'''
x = tf.placeholder(tf.float32, [None, 784]) placeholder 를 784로 처음에 준이유는 뭔가요?
처음 이미지 사이즈를 확인 하는 방법은 없나요?

>>>28x28=784
'''
f, a = plt.subplots(1, 10, figsize=(10, 2))
for i in range(10):
    a[i].imshow(np.reshape(mnist.test.images[i],(28,28)))
f.show()
plt.draw()
plt.waitforbuttonpress()
