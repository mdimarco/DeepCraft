DATA_DIR = "../../data/"
TRAIN_FILE = DATA_DIR + "train_images.txt"
TEST_FILE = DATA_DIR + "test_images.txt"

import numpy
from scipy import misc
from PIL import Image
from skimage import filters


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
		img = Image.fromarray(numpy.uint8(img))#.convert('L')
		img = img.resize((256, 256), Image.ANTIALIAS)

		#img = filters.sobel(img)
		img = numpy.asarray(img)
		# Flatten the image
		return img.reshape(img.size)
	images = map(image_helper, image_paths)
	images = numpy.asarray(images)
	
	return images, labels

def load_train_data():
	"""Returns a list of tuples (image, label) of the training images.
	These are respectively a 3d numpy array and a one hot label vector."""
	return _load_images(TRAIN_FILE)

def load_test_data():
	"""Returns a list of tuples (image, label) of the testing images.
	These are respectively a 3d numpy array and a one hot label vector."""
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

	return numpy.asarray(labels)

	