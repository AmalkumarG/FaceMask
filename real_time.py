from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


"""import dlib
import scipy.misc
from model import get_face_encodings, find_match



# tolerence of image..or confidence level
#TOLERANCE = 0.6


#loading databse...
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('dataset/'))

# Sorting images...
image_filenames = sorted(image_filenames)

#Generating Paths to images.. 
paths_to_images = ['dataset/' + x for x in image_filenames]


#generating face encodings of all images in database to face_encodings..
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

result=set()"""


"""def send_mail():
	sender_email = 'finalprojectmsc2021@gmail.com'
	receiver_email = 'amalkumarg1399@gmail.com'
	password = 'maskdetector'

	print("Sending the email...")

	server = smtplib.SMTP('smtp.gmail.com', 587)
	context = ssl.create_default_context()
	server.starttls(context=context)
	server.login(sender_email, password)

	msg = MIMEMultipart()		
	img_data = open('out.jpg', 'rb').read()
	text = MIMEText("no mask detected")
	msg.attach(text)
	image = MIMEImage(img_data, name=os.path.basename('out.jpg'))
	msg.attach(image)
			
	server.sendmail(sender_email, receiver_email, msg.as_string())
	print('Email sent!')

	print('Closing the server...')
	server.quit()"""

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)
	
faceNet = cv2.dnn.readNet("face_detector/deploy.prototxt.txt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model('mask_detector.model')

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame1=frame
    
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		#print("f1")
        
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
		label1 = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		print(label)
        
		cv2.putText(frame1, label1, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame1, (startX, startY), (endX, endY), color, 2)
		if label == "No Mask":
			#print("f2")
			cv2.imwrite("test\\test.jpg", frame)
			os.system("python face_rec.py")
			#cv2.imwrite("out.jpg", frame)
			#send_mail()
        
	cv2.imshow("Frame", frame1)
	#key = cv2.waitKey(1) & 0xFF
	cv2.waitKey(1)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		break

vs.stop()
vs.stream.release()
cv2.destroyAllWindows()

file=open("out.txt","r")
out=file.read()
file.close()
print("Result:",out)
message='Alert, '+out+' is not wearing mask'
print("message:",message)

from urllib.parse import urlencode
import pycurl

crl = pycurl.Curl()
crl.setopt(crl.URL, 'https://alc-training.in/gateway.php')
data = {'email': 'amalkumarg1399@gmail.com','msg':message}
pf = urlencode(data)

# Sets request method to POST,
# Content-Type header to application/x-www-form-urlencoded
# and data to send in request body.
crl.setopt(crl.POSTFIELDS, pf)
crl.perform()
crl.close()