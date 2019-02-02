import cv2
import matplotlib.pyplot as plt
import numpy as np

barrel_blue_mean = np.load('trained_parameters/barrel_blue_mean.npy')
barrel_blue_cov = np.load('trained_parameters/barrel_blue_cov.npy')
black_mean = np.load('trained_parameters/black_mean.npy')
black_cov = np.load('trained_parameters/black_cov.npy')
green_mean = np.load('trained_parameters/green_mean.npy')
green_cov = np.load('trained_parameters/green_cov.npy')
red_mean = np.load('trained_parameters/red_mean.npy')
red_cov = np.load('trained_parameters/red_cov.npy')
white_mean = np.load('trained_parameters/white_mean.npy')
white_cov = np.load('trained_parameters/white_cov.npy')
yellow_mean = np.load('trained_parameters/yellow_mean.npy')
yellow_cov = np.load('trained_parameters/yellow_cov.npy')
target_blue_mean = np.load('trained_parameters/target_blue_mean.npy')
target_blue_cov = np.load('trained_parameters/target_blue_cov.npy')
kettle_blue_mean = np.load('trained_parameters/kettle_blue_mean.npy')
kettle_blue_cov = np.load('trained_parameters/kettle_blue_cov.npy')
stick_blue_mean = np.load('trained_parameters/stick_blue_mean.npy')
stick_blue_cov = np.load('trained_parameters/stick_blue_cov.npy')
wall_blue_mean = np.load('trained_parameters/wall_blue_mean.npy')
wall_blue_cov = np.load('trained_parameters/wall_blue_cov.npy')
carpet_blue_mean = np.load('trained_parameters/carpet_blue_mean.npy')
carpet_blue_cov = np.load('trained_parameters/carpet_blue_cov.npy')


for l in range(46):
	print('Processing image# ' + str(l+1))
	#current_image = np.reshape(np.asarray(cv2.imread('trainset/' + str(l+1) + '.png')),(960000,3))
	#current_image = np.zeros((800,1200,2))
	current_image = np.asarray(cv2.imread('trainset/' + str(l+1) + '.png'))
	#current_image = np.asarray(cv2.cvtColor(cv2.imread('trainset/' + str(l+1) + '.png'), cv2.COLOR_BGR2HSV))
	current_image = np.reshape(current_image,(960000,3))
	mask_img = np.zeros((800,1200))
	scores = np.zeros((800,1200,6))
	barrel_blue_score = np.log(abs(np.linalg.det(barrel_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-barrel_blue_mean.transpose(),np.linalg.inv(barrel_blue_cov))),current_image.transpose()-barrel_blue_mean),axis=0),(800,1200))
	black_score = np.log(abs(np.linalg.det(black_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-black_mean.transpose(),np.linalg.inv(black_cov))),current_image.transpose()-black_mean),axis=0),(800,1200))
	green_score = np.log(abs(np.linalg.det(green_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-green_mean.transpose(),np.linalg.inv(green_cov))),current_image.transpose()-green_mean),axis=0),(800,1200))
	red_score = np.log(abs(np.linalg.det(red_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-red_mean.transpose(),np.linalg.inv(red_cov))),current_image.transpose()-red_mean),axis=0),(800,1200))
	white_score = np.log(abs(np.linalg.det(white_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-white_mean.transpose(),np.linalg.inv(white_cov))),current_image.transpose()-white_mean),axis=0),(800,1200))
	yellow_score = np.log(abs(np.linalg.det(yellow_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-yellow_mean.transpose(),np.linalg.inv(yellow_cov))),current_image.transpose()-yellow_mean),axis=0),(800,1200))
	scores[:,:,0] = barrel_blue_score
	scores[:,:,1] = black_score
	scores[:,:,2] = green_score
	scores[:,:,3] = red_score
	scores[:,:,4] = white_score
	scores[:,:,5] = yellow_score
	scores_m = np.argmin(scores,axis=2)
	mask_img[np.where(scores_m==0)] = 1
	#kernel = np.ones((5,5))
	#mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)




	current_image = np.asarray(cv2.cvtColor(cv2.imread('trainset/' + str(l+1) + '.png'), cv2.COLOR_BGR2HSV))
	#current_image = np.asarray(cv2.imread('trainset/' + str(l+1) + '.png'))
	v_channel = current_image[:,:,2]
	average_illuminance = np.mean(np.mean(v_channel[np.where(mask_img==1)]))
	print(average_illuminance)
	#current_image = current_image[:,:,0:2] #drop the v info
	current_image = np.reshape(current_image,(960000,3))
	target_blue_score = np.log(abs(np.linalg.det(target_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-target_blue_mean.transpose(),np.linalg.inv(target_blue_cov))),current_image.transpose()-target_blue_mean),axis=0),(800,1200))
	#not_target_blue_score = math.log(abs(np.linalg.det(not_target_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-not_target_blue_mean.transpose(),np.linalg.inv(not_target_blue_cov))),current_image.transpose()-not_target_blue_mean),axis=0),(800,1200))
	kettle_blue_score = np.log(abs(np.linalg.det(kettle_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-kettle_blue_mean.transpose(),np.linalg.inv(kettle_blue_cov))),current_image.transpose()-kettle_blue_mean),axis=0),(800,1200))
	stick_blue_score = np.log(abs(np.linalg.det(stick_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-stick_blue_mean.transpose(),np.linalg.inv(stick_blue_cov))),current_image.transpose()-stick_blue_mean),axis=0),(800,1200))
	wall_blue_score = np.log(abs(np.linalg.det(wall_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-wall_blue_mean.transpose(),np.linalg.inv(wall_blue_cov))),current_image.transpose()-wall_blue_mean),axis=0),(800,1200))
	carpet_blue_score = np.log(abs(np.linalg.det(carpet_blue_cov))) + np.reshape(np.sum(np.multiply(np.transpose(np.dot(current_image-carpet_blue_mean.transpose(),np.linalg.inv(carpet_blue_cov))),current_image.transpose()-carpet_blue_mean),axis=0),(800,1200))
	secondscores = np.zeros((800,1200,5))
	secondscores[:,:,0] = target_blue_score
	secondscores[:,:,1] = kettle_blue_score
	secondscores[:,:,2] = stick_blue_score
	secondscores[:,:,3] = wall_blue_score
	secondscores[:,:,4] = carpet_blue_score
	#secondscores[:,:,1] = target_blue_score+10
	#secondscores[:,:,2] = target_blue_score+10
	#secondscores[:,:,3] = target_blue_score+10
	#secondscores[:,:,4] = target_blue_score+10
	if average_illuminance < 50:
		print(1)
	else:
		secondscores_m = np.argmin(secondscores,axis=2)
		secondscores_m[np.where(mask_img==0)] = 1
		mask_img[np.where(secondscores_m!=0)] = 0
	
	#kernel = np.ones((3,3))
	# mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
	#mask_img = cv2.dilate(mask_img,kernel,iterations = 1)
	#mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
	plt.imsave('second_classified_results/'+str(l+1)+'.png',mask_img,cmap='gray')




