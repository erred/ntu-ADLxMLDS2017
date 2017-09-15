from mnist_reader import load_mnist

import tensorflow as tf
import numpy as np

X_train, y_train = load_mnist('data/fashion', kind='train')
X_test, y_test = load_mnist('data/fashion', kind='t10k')

y_train2 = np.zeros((y_train.size, 10))
y_train2[np.arange(y_train.size), y_train] = 1

y_test2 = np.zeros((y_test.size, 10))
y_test2[np.arange(y_test.size), y_test] = 1

# images go here
x = tf.placeholder(tf.float32, [None, 784])
# Weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(2000):
  # batch_xs, batch_ys = mnist.t
  sess.run(train_step, feed_dict={x: X_train, y_: y_train2})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X_test, y_: y_test2}))
