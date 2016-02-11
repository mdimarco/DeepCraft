import numpy
from matplotlib import pyplot as plt 
from skimage import io, filters 
from PIL import Image
from scipy import misc


# Load the images from their paths
# and apply filter to said image
def apply_filter(path, frequency, theta):

	img = misc.imread(path)
	# HACK - MAJOR HACK - this forces all images to be the same size
	img = Image.fromarray(numpy.uint8(img)).convert('L')
	img_resized = img.resize((256, 256), Image.ANTIALIAS)

	#plt.figure()
	#plt.title('Raw Image')
	#io.imshow(numpy.asarray(img) )

	img_filtered = filters.gabor_filter(img_resized, frequency=frequency, theta=theta)
	#img = numpy.asarray(img)
	# Flatten the image
	return img_filtered


path = '../../data/processed'


for fname in ['/cactus/cactus4/cactus_4.mov.25.png','/cactus/cactus_3/cactus_3-3.png','/cactus/cactus5/cactus_5.mov.38.png']:
	for freq in [.8]:
		for theta in [20,45,90]:

			img_filtered = apply_filter( path+fname ,freq,theta)
			plt.figure()
			plt.title('Gabor Filter Freq:'+str(freq)+" Theta:"+str(theta))
			io.imshow(img_filtered[0])


io.show()



