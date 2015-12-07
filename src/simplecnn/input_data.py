DATA_DIR = "../../data/"
TRAIN_FILE = DATA_DIR + "train_images.txt"
TEST_FILE = DATA_DIR + "test_images.txt"

import numpy
from scipy import misc
from PIL import Image

def _extract_labels(image_path_list):
	"""Extracts the labels for a list of images given the paths to the images"""
	def extract_label(x):
		return x.split('/')[-3] # In reverse it goes image name, video name, video label
	return map(extract_label, image_path_list)

def _load_images(image_path_file):
	"""Loads the training data from disk, storing the images in memory.
	Returns (images, labels) where images are numpy arrays of the images and
	labels are one-hot numpy arrays representing the labels."""
	# Load the image file paths
	with open(image_path_file, 'r') as f:
		lines = list(f)

	# Remove the whitespace at the end of any lines and prepend the data dir
	def map_helper(x):
		return DATA_DIR+x.rstrip()
	image_paths = map(map_helper, lines)
	labels = _extract_labels(lines)
	labels = dense_to_one_hot(labels)

	# Load the images from their paths
	def image_helper(x):
		img = misc.imread(x)
		# HACK - MAJOR HACK - this forces all images to be the same size
		img = Image.fromarray(numpy.uint8(img))
		#img = img.convert('L').resize((28, 28), Image.ANTIALIAS)
		img = img.resize((128, 128), Image.ANTIALIAS)
		img = numpy.asarray(img)
		# Flatten the image
		img = img.reshape(img.size)
		img = img.astype(numpy.float32)
		img = numpy.multiply(img, 1.0 / 255.0)
		return img
	images = map(image_helper, image_paths)
	images = numpy.asarray(images)
	
	return images, labels

def load_train_data():
	"""Returns a tuple of lists (image, label) of the training images and labels.
	These are respectively a list of 3d numpy arrays and a list of one hot label vectors."""
	return _load_images(TRAIN_FILE)

def load_test_data():
	"""Returns a tuple of lists (image, label) of the testing images and labels.
	These are respectively a list of 3d numpy arrays and a list of one hot label vectors."""
	return _load_images(TEST_FILE)

def dense_to_one_hot(labels):
	"""Convert class labels from text to one-hot vectors."""

	# First, determine the number of classes
	label_list = list(set(labels))
	num_classes = len(label_list)

	# Convert the labels to numbers
	labels = labels[:]
	for i, label in enumerate(label_list):
		labels = [i if x==label else x for x in labels]

	labels_dense = numpy.asarray(labels)

	num_labels = labels_dense.shape[0]
	index_offset = numpy.arange(num_labels) * num_classes
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

