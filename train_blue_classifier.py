import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


labels = ['target_blue','kettle_blue','stick_blue','wall_blue','carpet_blue']
#numofpicture = 46
numofclass = 5


for l in range(numofclass):
	label = labels[l]
	folder = 'blue_trainset/'+label
	M = np.empty((3,1))
	for filename in os.listdir(folder+'/image'):
		# read one test image
		#img = cv2.imread(os.path.join(folder,filename))
	#for x in range(numofpicture):
		#print('Processing image #' + str(x+1) + ' of label: ' + label)
		print('Processing ' + filename + ' of label: ' + label)
		#MASKPATH = 'blue_trainset/blue_masks/' + label + '/' + str(x+1) + '.png'
		MASKPATH = folder+'/mask/' + filename
		#IMAGEPATH = 'blue_trainset/' + str(x+1) + '.png'
		IMAGEPATH = folder+'/image/' + filename
		image = np.asarray(cv2.cvtColor(cv2.imread(IMAGEPATH), cv2.COLOR_BGR2HSV))
		#image = np.asarray(cv2.imread(IMAGEPATH))
		#image = image[:,:,0:2] #drop the v info
		# image = np.zeros((800,1200,4))
		# image[:,:,0:3] = np.asarray(cv2.imread(IMAGEPATH))
		# image[:,:,3] = np.asarray(cv2.cvtColor(cv2.imread(IMAGEPATH), cv2.COLOR_BGR2HSV))[:,:,2]
		mask = np.asarray(plt.imread(MASKPATH))
		M = np.hstack((M,image[np.where(mask==1)].transpose()))
	M_mean = np.mean(M, axis=1).reshape(3,1)
	M_cov = np.cov(M)
	np.save('trained_parameters/'+ label + '_mean.npy', M_mean)
	np.save('trained_parameters/'+ label + '_cov.npy', M_cov)