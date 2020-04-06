from PIL import Image
import tensorflow as tf
import numpy as np

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784], name='input_x') / 255.  # 28x28
ys = tf.placeholder(tf.float32, [None, 10], name='input_y')
keep_prob = tf.placeholder(tf.float32)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'myModelC/model.ckpt')

    print(sess.run(tf.argmax(train_step, 1), {input_x: mnist.train.images[0:10, :]}))
    print(sess.run(tf.argmax(mnist.train.labels[0:10, :], 1)))



