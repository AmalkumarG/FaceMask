import dlib
import scipy.misc
import numpy as np
import os
import cv2
from model import get_face_encodings, find_match


#To detects face from the image...
#face_detector = dlib.get_frontal_face_detector()

#TO detect landamark points and pose/angle in the face
#shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#To generate the face encodings from the image
#face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')



# tolerence of image..or confidence level
#TOLERANCE = 0.6


#loading criminal databse...
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('dataset/'))

# Sorting images...
image_filenames = sorted(image_filenames)

#Generating Paths to images.. 
paths_to_images = ['dataset/' + x for x in image_filenames]


#generating face encodings of all images in criminal database to face_encodings..
face_encodings = []

for path_to_image in paths_to_images:
  
    face_encodings_in_image = get_face_encodings(path_to_image)
    
	#only one face in an image accepted..
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    face_encodings.append(get_face_encodings(path_to_image)[0])
'''
f=open("input.txt","r")
inp=f.read()	
f.close()'''
	
#loading test image..
#test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))

#Generating Paths to images.. 
#paths_to_test_images = ['test/' + x for x in test_filenames]

#Generating names from Imagefilename..
names = [x[:-4] for x in image_filenames]

result=set()
"""
cap = cv2.VideoCapture("input\\test.mp4")
#cap = cv2.VideoCapture("test1.mp4")
#Comparing test images with criminal database one by one....
while True:  
    # Read the frame  
	ret, img = cap.read()  
	
	if not ret: # New
		break 
	
	cv2.imwrite("D:\\WORK_SPACE\\face_recognition\\deliver\\test\\test.jpg", img)"""
	
path_name="test\\test.jpg"
	
face_encodings_in_image = get_face_encodings(path_name)
    
	#only one face in an image accepted..
'''
	if len(face_encodings_in_image) != 1:
		print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
		continue
		#exit()'''
	
	#cheking similarity with each image in database..
for i in range(len(face_encodings_in_image)):
	res = find_match(face_encodings, names, face_encodings_in_image[i])
	result.add(str(res))
	#print(res)
	'''
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
  
    # Display  
    cv2.imshow('Video', img)  '''
  
    # Stop if escape key is pressed  
"""	k = cv2.waitKey(30) & 0xff  
	if k==27:  
		break  
cap.release()
cv2.destroyAllWindows()"""

result=list(result)
		
print("final result:",result[0])


f=open("out.txt","w")
f.write(result[0])	
f.close()

	
'''		
for path_to_image in paths_to_test_images:

    #generating test face encodings...
	face_encodings_in_image = get_face_encodings(path_to_image)
	
    #only one face in an image accepted..
	if len(face_encodings_in_image) != 1:
		print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
		exit()
		
	#cheking similarity with each image in database..
	res = find_match(face_encodings, names, face_encodings_in_image[0])
	
	file =open("out.txt","w")
	file.write(res)
	file.close()
	print("Checking crime database:",res)'''
