#

# The number of steps to take for training
# NOTE: This got 0.187817 accuracy with 100 steps and no shuffling and 640x640 images
# And 0.187817 for 50 steps and shuffling with 320x320 images, and same with 10 steps
# Same accuracy with 320x320 1000 steps
# Gets 0.20202 accuracy with 5000+ steps with testing the data it trained on, so something is borked
# 
# With the simple cnn and two classes (cactus vs desert no object)
# and 28*2 x 28*2 images, we get 0.856164 accuracy
# 
# 
TRAIN_STEPS = 500
BATCH_SIZE = 50

import input_data

# For MINST
'''
import input_data_minst
mnist = input_data_minst.read_data_sets('MNIST_data', one_hot=True)
train_images, train_labels = mnist.train.images, mnist.train.labels
test_images, test_labels = mnist.test.images, mnist.test.labels
'''
##

print "Loading data"

train_images, train_labels = input_data.load_train_data()
test_images, test_labels = input_data.load_test_data()

FLAT_IMG_SIZE = train_images[0].shape[0]
NUM_CLASSES = train_labels.shape[1]
print "Data loaded"


# Show an image
#import matplotlib.pyplot as plt
#plt.imshow(train_images[0].reshape(320, 320), cmap=plt.cm.gray)

import numpy
import tensorflow as tf
# Hack - should not use interactive session for fixed code
sess = tf.InteractiveSession()

'''
# This image size of 640x640 is enforced as a hack in input_data.py
x = tf.placeholder("float", shape=[None, FLAT_IMG_SIZE])
y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

W = tf.Variable(tf.zeros([FLAT_IMG_SIZE,NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
for i in range(TRAIN_STEPS):
	if i%10 == 0:
		print "Train step:", i

	# Permute the data each run with numpy
	perm = numpy.arange(train_images.shape[0])
	numpy.random.shuffle(perm)
	train_images = train_images[perm]
	train_labels = train_labels[perm]

	train_step.run(feed_dict={x: train_images[:50], y_: train_labels[:50]})
print "Done training"

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "Testing"

print accuracy.eval(feed_dict={x: test_images, y_: test_labels})
'''

# From CIFAR-10
# 

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable

	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	var = _variable_on_cpu(name, shape,
												 tf.truncated_normal_initializer(stddev=stddev))
	if wd:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def inference(images):
	"""Build the CIFAR-10 model.

	Args:
		images: Images returned from distorted_inputs() or inputs().

	Returns:
		Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	#
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
											stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
												 padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
										name='norm1')

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
												stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
						name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
							strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		dim = 1
		for d in pool2.get_shape()[1:].as_list():
			dim *= d
		reshape = tf.reshape(pool2, [BATCH_SIZE, dim])

		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
																					stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
																					stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
																					stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
															tf.constant_initializer(0.0))
		softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

	return softmax_linear



# From tutorial

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, FLAT_IMG_SIZE])
y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

# First layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Warning: assumes square images
from math import sqrt
IMG_HEIGHT = int(sqrt(FLAT_IMG_SIZE))
IMG_WIDTH = IMG_HEIGHT

x_image = tf.reshape(x, [-1,IMG_HEIGHT,IMG_WIDTH,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print "Calculating dimensions"
dim = 1
for d in h_pool1.get_shape()[1:].as_list():
	print d
	dim *= d
print dim

h_pool1_safe = tf.verify_tensor_all_finite(h_pool1, "Failed on 1", name=None)
tf.add_check_numerics_ops()
# Second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1_safe, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Dense layer (fully connected)
print "Calculating dimensions"
dim = 1
for d in h_pool2.get_shape()[1:].as_list():
	print d
	dim *= d
print dim

W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout to prevent overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout (softmax)
W_fc2 = weight_variable([1024, NUM_CLASSES])
b_fc2 = bias_variable([NUM_CLASSES])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Train
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(TRAIN_STEPS):
	# Permute the data each run with numpy
	perm = numpy.arange(train_images.shape[0])
	numpy.random.shuffle(perm)
	train_images = train_images[perm]
	train_labels = train_labels[perm]

	batch = (train_images[:BATCH_SIZE], train_labels[:BATCH_SIZE])
	#print "Step", i
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch[0], y_: batch[1], keep_prob: 1.0})
		print "step %d, training accuracy %g"%(i, train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: test_images, y_: test_labels, keep_prob: 1.0})
