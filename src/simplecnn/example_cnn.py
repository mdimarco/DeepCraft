
# On cactus vs desert no object, with 28x28 images, this gets 0.98630136986301364 accuracy
# 
# On bunny, cactus, chicken, cow, desert_noobject, forest_noobject, pig, sheep, tree in color at 28*28
# we get 0.783819628647 accuracy (9 classes) after 500 training steps,
# 0.809018567639 after 1000 training steps
# 
# For 128x128, we get between 0.74 and 0.8 accuracy for all classes
import input_data

TRAIN_STEPS = 500
BATCH_SIZE = 50


MNIST = False
# For MINST
if MNIST:
	import input_data_minst
	mnist = input_data_minst.read_data_sets('MNIST_data', one_hot=True)
	train_images, train_labels = mnist.train.images, mnist.train.labels
	test_images, test_labels = mnist.test.images, mnist.test.labels
else:
	print "Loading data"

	train_images, train_labels = input_data.load_train_data()
	test_images, test_labels = input_data.load_test_data()

FLAT_IMG_SIZE = train_images[0].shape[0]
NUM_CLASSES = train_labels.shape[1]
print "Data loaded"


import numpy as np
import tensorflow as tf

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
	l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
	l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
	l1 = tf.nn.dropout(l1, p_keep_conv)

	l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
	l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
	l2 = tf.nn.dropout(l2, p_keep_conv)

	l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
	l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
	l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
	l3 = tf.nn.dropout(l3, p_keep_conv)

	l4 = tf.nn.relu(tf.matmul(l3, w4))
	l4 = tf.nn.dropout(l4, p_keep_hidden)

	pyx = tf.matmul(l4, w_o)
	return pyx

# Warning: assumes square images
from math import sqrt
IMG_HEIGHT = 128#int(sqrt(FLAT_IMG_SIZE))
IMG_WIDTH = IMG_HEIGHT

# Reshape images
train_images = train_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)
test_images = test_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)

# Input variables
X = tf.placeholder('float', [None, IMG_HEIGHT, IMG_WIDTH, 3])
Y = tf.placeholder('float', [None, NUM_CLASSES])

# Weights
w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 16 * 16, 625])
w_o = init_weights([625, NUM_CLASSES])

p_keep_conv = tf.placeholder('float')
p_keep_hidden = tf.placeholder('float')

my_model = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(my_model, Y))

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(my_model, 1)

# Start the session and init all variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Train
for i in range(TRAIN_STEPS):
	print "Run", i
	for start, end in zip(range(0, len(train_images), 128), range(128, len(train_images), 128)):
		sess.run(train_op, feed_dict={X: train_images[start:end], Y: train_labels[start:end],
									  p_keep_conv: 0.8, p_keep_hidden: 0.5})
	
	if i % 10 == 0:
		test_indices = np.arange(len(test_images)) # Get A Test Batch
		np.random.shuffle(test_indices)
		test_indices = test_indices[0:50]
		
		print i, np.mean(np.argmax(test_labels[test_indices], axis=1) ==
						 sess.run(predict_op, feed_dict={X: test_images[test_indices],
														 Y: test_labels[test_indices],
														 p_keep_conv: 1.0,
														 p_keep_hidden: 1.0}))

print "Accuracy on all test images:", np.mean(np.argmax(test_labels, axis=1) == sess.run(predict_op, feed_dict={X: test_images,Y: test_labels,p_keep_conv: 1.0,p_keep_hidden: 1.0}))
