import cv2, os
import numpy as np
from PIL import Image
import json

faceCascade = cv2.CascadeClassifier("/home/jain/Downloads/opencv-2.4.5/data/haarcascades/haarcascade_frontalface_default.xml");

#for recognizing
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path): 
	# Append all the absolute image paths in a list image_paths 
	# We will not read the image with the .sad extension in the training set 
	# Rather, we will use them to test our accuracy of the training 
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('low.JPG')] 
	# images will contains face images 
	images = [] 
	# labels will contains the label that is assigned to the image 
	labels = [] 
	#print image_paths
	for image_path in image_paths: 
		# Read the image and convert to grayscale 
		image_pil = Image.open(image_path).convert('L') 
		# Convert the image format into numpy array 
		image = np.array(image_pil, 'uint8') 
		# Get the label of the image 
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", "")) 
		# Detect the face in the image 
		faces = faceCascade.detectMultiScale(image) 
		# If face is detected, append the face to images and the label to labels 
		for (x, y, w, h) in faces: 
			images.append(image[y: y + h, x: x + w]) 
			labels.append(nbr) 
			#print images
			cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w]) 
			cv2.waitKey(50) 
	# return the images list and labels list 
	return images, labels
path = './faces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the training
recognizer.train(images, np.array(labels))
path = './faces-master'
# Append the images with the extension .sad into image_paths 
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPG')] 

for image_path in image_paths: 
	print image_path
	predict_image_pil = Image.open(image_path).convert('L') 
	predict_image = np.array(predict_image_pil, 'uint8') 
	faces = faceCascade.detectMultiScale(predict_image) 
	for (x, y, w, h) in faces: 
		nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w]) 
		#print nbr_predicted
		nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", "")) 
		#print nbr_actual
		
		#JSON file fetching
		
		with open('face-map.json') as datas:
			data = json.load(datas)
		if conf < 100:
			print data['face'][str(nbr_predicted)]
			print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
		

		if nbr_actual == nbr_predicted: 
			pass	
		#print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf) 
		#else: 
		#print "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted) 
		cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w]) 
		cv2.waitKey(1000)


