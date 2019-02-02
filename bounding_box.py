import cv2
import matplotlib.pyplot as plt
import numpy as np

for l in range(46):
	print('Processing image# '+str(l+1))
	IMAGEPATH = 'first_classified_results/' + str(l+1) + '.png'
	img = cv2.imread(IMAGEPATH)
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)


	for i in range(np.shape(contours)[0]):
		if (contours[i].size>100):
			x,y,w,h = cv2.boundingRect(contours[i])
			if h > 0.8*w and h < 2.5*w:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	img.dtype='uint8'
	cv2.imwrite('bounding_box_results/'+ str(l+1) + '.png', img)
	# cv2.imshow('drawimg',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()