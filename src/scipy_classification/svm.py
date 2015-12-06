

import numpy
from scipy import misc
from sklearn import svm
from sklearn import cross_validation

from skimage import io, filters
from PIL import Image

import input_data

#Loading data from input_data script
train_images,train_labels = input_data.load_train_data()
test_images,test_labels = input_data.load_test_data()



#Train SVM
classifier = svm.SVC(C=.000001)
classifier.fit(train_images, train_labels)

train_score = classifier.score(train_images, train_labels)
train_xval_score = cross_validation.cross_val_score(classifier,train_images,train_labels,cv=10,scoring='accuracy')
test_score = classifier.score(test_images, test_labels)

print("Train score:",train_score)
print("Train xval score:",train_xval_score)
print("Test score",test_score)