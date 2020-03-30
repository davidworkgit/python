import numpy as np
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# create tensorflow structure end
sess = tf.Session()
sess.run(init)

for setp in range(201):
    sess.run(train)
    if setp % 20 == 0:
        print(setp, sess.run(Weights), sess.run(biases))
