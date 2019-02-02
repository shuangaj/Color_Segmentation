import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


labels = ['barrel_blue','black','green','red','white','yellow']
#numofpicture = 46
numofclass = 6

for l in range(numofclass):
	label = labels[l]
	M = np.empty((3,1))
	folder = 'general_trainset/'+label
	for filename in os.listdir(folder+'/image'):
		print('Processing ' + filename + ' of label: ' + label)
		MASKPATH = folder+'/mask/' + filename
		IMAGEPATH = folder+'/image/' + filename
		image = np.asarray(cv2.imread(IMAGEPATH))
		#image = np.zeros((800,1200,3))
		#image[:,:,0:3] = np.asarray(cv2.imread(IMAGEPATH))
		#image = np.asarray(cv2.cvtColor(cv2.imread(IMAGEPATH), cv2.COLOR_BGR2HSV))
		mask = np.asarray(plt.imread(MASKPATH))
		M = np.hstack((M,image[np.where(mask==1)].transpose()))
	M_mean = np.mean(M, axis=1).reshape(3,1)
	M_cov = np.cov(M)
	np.save('trained_parameters/'+ label + '_mean.npy', M_mean)
	np.save('trained_parameters/'+ label + '_cov.npy', M_cov)