"""Reads in all of our data and selects a training and testing set

Videos will not be split so that if one image of a video is in the training set,
no image from that video will be in the testing set.
This avoids training and testing on highly correlated data."""

# The percentage of data to use for training
TRAIN_PERCENT = 0.5
# Path to the processed data folder
ROOT_DIR = "./processed"
# Path to the directory to save the output files
OUTFILE_DIR = "."


import os
import math

from collections import defaultdict

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Each subdirectory of the root directory is a label
labels = get_subdirectories(ROOT_DIR)

train_images = list()
test_images = list()

for label in labels:
	label_root_path = ROOT_DIR+"/"+label
	# Each video is a folder
	videos = get_subdirectories(label_root_path)
	
	# Select a subset of videos for training
	cutoff_index = int( math.ceil( len(videos)*TRAIN_PERCENT ) )

	# Select nonoverlapping subsets of the list
	train_subset = videos[:cutoff_index]
	test_subset = videos[cutoff_index:]

	train_subset_images, test_subset_images = list(), list()

	# Loop over the training subset of videos
	for video_name in train_subset:
		video_path = label_root_path+'/'+video_name
		images = [filename for filename in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, filename)) and filename[0] != '.']
		for image_name in images:
			train_subset_images.append(video_path+"/"+image_name)

	# Do the same for the testing subset
	for video_name in test_subset:
		video_path = label_root_path+'/'+video_name
		images = [filename for filename in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, filename)) and filename[0] != '.']
		for image_name in images:
			test_subset_images.append(video_path+"/"+image_name)


	# Add the training and testing data to their respective lists in the
	# training and testing data lists
	train_images.extend(train_subset_images)
	test_images.extend(test_subset_images)

# Add newlines to the end of all the file names for printing
train_images = map(lambda x: x+"\n", train_images)
test_images = map(lambda x: x+"\n", test_images)

# Save the list of training images
with open(OUTFILE_DIR+'/train_images.txt', 'w') as f:
	f.writelines(train_images)

# Save the list of testing images
with open(OUTFILE_DIR+'/test_images.txt', 'w') as f:
	f.writelines(test_images)

print "Done!"