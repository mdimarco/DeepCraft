#

# The number of steps to take for training
# NOTE: This got 0.187817 accuracy with 100 steps and no shuffling and 640x640 images
# And 0.187817 for 50 steps and shuffling with 320x320 images, and same with 10 steps
# Same accuracy with 320x320 1000 steps
# Gets 0.20202 accuracy with 5000+ steps with testing the data it trained on, so something is borked
TRAIN_STEPS = 20000

import input_data

print "Loading data"
train_images, train_labels = input_data.load_train_data()
test_images, test_labels = input_data.load_test_data()

FLAT_IMG_SIZE = train_images[0].shape[0]
NUM_CLASSES = train_labels.shape[1]
print "Data loaded"

import numpy
import tensorflow as tf
# Hack - should not use interactive session for fixed code
sess = tf.InteractiveSession()


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

	train_step.run(feed_dict={x: train_images, y_: train_labels})
print "Done training"

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "Testing"

print accuracy.eval(feed_dict={x: train_images, y_: train_labels})