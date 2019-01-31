'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np

class BarrelDetector():
	def __init__(self):
		self.barrel_blue_mean = np.load('trained_parameters/barrel_blue_mean.npy')
		self.barrel_blue_cov = np.load('trained_parameters/barrel_blue_cov.npy')
		self.black_mean = np.load('trained_parameters/black_mean.npy')
		self.black_cov = np.load('trained_parameters/black_cov.npy')
		self.green_mean = np.load('trained_parameters/green_mean.npy')
		self.green_cov = np.load('trained_parameters/green_cov.npy')
		self.red_mean = np.load('trained_parameters/red_mean.npy')
		self.red_cov = np.load('trained_parameters/red_cov.npy')
		self.white_mean = np.load('trained_parameters/white_mean.npy')
		self.white_cov = np.load('trained_parameters/white_cov.npy')
		self.yellow_mean = np.load('trained_parameters/yellow_mean.npy')
		self.yellow_cov = np.load('trained_parameters/yellow_cov.npy')
		self.target_blue_mean = np.load('trained_parameters/target_blue_mean.npy')
		self.target_blue_cov = np.load('trained_parameters/target_blue_cov.npy')
		self.not_target_blue_mean = np.load('trained_parameters/not_target_blue_mean.npy')
		self.not_target_blue_cov = np.load('trained_parameters/not_target_blue_cov.npy')

	def segment_image(self, img):
		current_image = np.zeros((800,1200,4))
		current_image[:,:,0:3] = np.asarray(img)
		current_image[:,:,3] = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))[:,:,0]
		current_image = np.reshape(current_image,(960000,4))
		mask_img = np.zeros((800,1200))
		scores = np.zeros((800,1200,6))
		barrel_blue_score = np.log(abs(np.linalg.det(self.barrel_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.barrel_blue_mean.transpose(),np.linalg.inv(self.barrel_blue_cov))),current_image.transpose()-self.barrel_blue_mean),axis=0),(800,1200))
		black_score = np.log(abs(np.linalg.det(self.black_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.black_mean.transpose(),np.linalg.inv(self.black_cov))),current_image.transpose()-self.black_mean),axis=0),(800,1200))
		green_score = np.log(abs(np.linalg.det(self.green_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.green_mean.transpose(),np.linalg.inv(self.green_cov))),current_image.transpose()-self.green_mean),axis=0),(800,1200))
		red_score = np.log(abs(np.linalg.det(self.red_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.red_mean.transpose(),np.linalg.inv(self.red_cov))),current_image.transpose()-self.red_mean),axis=0),(800,1200))
		white_score = np.log(abs(np.linalg.det(self.white_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.white_mean.transpose(),np.linalg.inv(self.white_cov))),current_image.transpose()-self.white_mean),axis=0),(800,1200))
		yellow_score = np.log(abs(np.linalg.det(self.yellow_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.yellow_mean.transpose(),np.linalg.inv(self.yellow_cov))),current_image.transpose()-self.yellow_mean),axis=0),(800,1200))
		scores[:,:,0] = barrel_blue_score
		scores[:,:,1] = black_score
		scores[:,:,2] = green_score
		scores[:,:,3] = red_score
		scores[:,:,4] = white_score
		scores[:,:,5] = yellow_score
		scores_m = np.argmin(scores,axis=2)
		mask_img[np.where(scores_m==0)] = 1
		kernel = np.ones((3,3))
		mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)


		current_image = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
		current_image = np.reshape(current_image,(960000,3))
		target_blue_score = np.log(abs(np.linalg.det(self.target_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.target_blue_mean.transpose(),np.linalg.inv(self.target_blue_cov))),current_image.transpose()-self.target_blue_mean),axis=0),(800,1200))
		not_target_blue_score = np.log(abs(np.linalg.det(self.not_target_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-self.not_target_blue_mean.transpose(),np.linalg.inv(self.not_target_blue_cov))),current_image.transpose()-self.not_target_blue_mean),axis=0),(800,1200))
		secondscores = np.zeros((800,1200,2))
		secondscores[:,:,0] = target_blue_score
		secondscores[:,:,1] = not_target_blue_score
		secondscores_m = np.argmin(secondscores,axis=2)
		secondscores_m[np.where(mask_img==0)] = 1
		mask_img[np.where(secondscores_m==1)] = 0
		
		kernel = np.ones((3,3))
		mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
		mask_img = cv2.dilate(mask_img,kernel,iterations = 2)
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		img = np.array(self.segment_image(img),np.uint8)*255
		# print(img.shape)
		ret,thresh = cv2.threshold(img,127,255,0)
		contours,hierarchy = cv2.findContours(thresh, 1, 2)
		boxes = []
		#print(np.shape(contours)[0])
		for i in range(np.shape(contours)[0]):
			if (contours[i].size>20):
				x,y,w,h = cv2.boundingRect(contours[i])
				if h > 0.8*w and h < 3*w:
					#print('yes')
					#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
					boxes.append([x,y,x+w,y+h])
		#cv2.imwrite('bounding_box_results/'+ str(1) + '.png', img)
		print(boxes)
		return boxes


if __name__ == '__main__':
	folder = "trainset"
	my_detector = BarrelDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

