

import numpy
from scipy import misc
from sklearn import svm
from sklearn import cross_validation

from skimage import io, filters
from matplotlib import pyplot as plt  
from PIL import Image

import input_data


global filtercounter
filtercounter = 0

# Load the images from their paths
# and apply filter to said image
def apply_filter(path, frequency, theta):
	global filtercounter
	filtercounter+=1
	print(frequency,theta,filtercounter)


	img = misc.imread(path)
	# HACK - MAJOR HACK - this forces all images to be the same size
	img = Image.fromarray(numpy.uint8(img)).convert('L')
	img = img.resize((10, 10), Image.ANTIALIAS)

	img = filters.gabor_filter(img, frequency=frequency, theta=theta)
	img = numpy.asarray(img)
	# Flatten the image
	return img.reshape(img.size)


def apply_filters(image_paths,labels):

	imgs = []
	# Need to extend labels to match extra images
	labels_extended = []
	for theta in [20,45,90]:
		for freq in [.8]:
			print("Running Gabor",theta,freq)
			imgs.extend( map(lambda x: apply_filter(x,freq,theta), image_paths) )
			labels_extended.extend( labels )
	
	return imgs,labels_extended




#Loading data from input_data script
train_images,train_labels = input_data.load_train_data(apply_filters)
test_images,test_labels = input_data.load_test_data(apply_filters)



#Train SVM
classifier = svm.SVC(C=.01)
classifier.fit(train_images, train_labels)

train_score = classifier.score(train_images, train_labels)
train_xval_score = cross_validation.cross_val_score(classifier,train_images,train_labels,cv=10,scoring='accuracy')
test_score = classifier.score(test_images, test_labels)

print("Short Gabors 20-45-90,.8")
print("Train score:",train_score)
print("Train xval score:",train_xval_score)
print("Test score",test_score)


# 1.

# Black and White
# ('Train score:', 0.67363344051446949)
# ('Train xval score:', array([ 0.66666667,  0.66666667,  0.66666667,  0.67741935,  0.67741935,
#         0.67741935,  0.67741935,  0.67741935,  0.67741935,  0.67213115]))
# ('Test score', 0.42622950819672129)

# Color
# ('Train score:', 0.67363344051446949)
# ('Train xval score:', array([ 0.66666667,  0.66666667,  0.66666667,  0.67741935,  0.67741935,
#         0.67741935,  0.67741935,  0.67741935,  0.67741935,  0.67213115]))
# ('Test score', 0.42622950819672129)

# Sobel
# ('Train score:', 0.67363344051446949)
# ('Train xval score:', array([ 0.66666667,  0.66666667,  0.66666667,  0.67741935,  0.67741935,
#         0.67741935,  0.67741935,  0.67741935,  0.67741935,  0.67213115]))
# ('Test score', 0.42622950819672129)


# 2. 
# 
# Black and White
# ('Train score:', 0.67395264116575593)
# ('Train xval score:', array([ 0.67272727,  0.67272727,  0.67272727,  0.67272727,  0.67272727,
#         0.67272727,  0.67272727,  0.67272727,  0.67272727,  0.68518519]))
# ('Test score', 0.517948717948718)
# 
# Sobel
# ('Train score:', 0.67395264116575593)
# ('Train xval score:', array([ 0.67272727,  0.67272727,  0.67272727,  0.67272727,  0.67272727,
#         0.67272727,  0.67272727,  0.67272727,  0.67272727,  0.68518519]))
# ('Test score', 0.517948717948718)
# 
# 
# 
# 3. 
# 
# Long gabors theta=[0,45,90], freq=[.1,2,3]
# ('Train score:', 0.64976228209191755)
# ('Train xval score:', array([ 0.64964789,  0.64964789,  0.64964789,  0.64964789,  0.64964789,
#         0.64964789,  0.64964789,  0.64964789,  0.64964789,  0.65079365]))
# ('Test score', 0.53982300884955747)
# 
# 
# Short Gabors theta=0, freq = .1,2,3
# ('Train score:', 0.64976228209191755)
# ('Train xval score:', array([ 0.64736842,  0.64736842,  0.64736842,  0.65079365,  0.65079365,
#         0.65079365,  0.65079365,  0.65079365,  0.65079365,  0.65079365]))
# ('Test score', 0.53982300884955747)
# 
# 
# Short Gabors 0-45,2
# ('Train score:', 0.64976228209191755)
# ('Train xval score:', array([ 0.64566929,  0.64566929,  0.65079365,  0.65079365,  0.65079365,
#         0.65079365,  0.65079365,  0.65079365,  0.65079365,  0.65079365]))
# ('Test score', 0.53982300884955747)
# 
# 
# 
# 
# # Short Gabors 0-45-90,.3
# ('Train score:', 0.64976228209191755)
# ('Train xval score:', array([ 0.64736842,  0.64736842,  0.64736842,  0.65079365,  0.65079365,
#         0.65079365,  0.65079365,  0.65079365,  0.65079365,  0.65079365]))
# ('Test score', 0.53982300884955747)