import dlib
import scipy.misc
import numpy as np
import os

names=['mark','bill']


face_detector = dlib.get_frontal_face_detector()

#TO detect landamark points and pose/angle in the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#To generate the face encodings from the image
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# tolerence of image..or confidence level
TOLERANCE = 0.6

def get_face_encodings(path_to_image):
    image = scipy.misc.imread(path_to_image,mode='RGB')
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces, face):
	return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)

def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return 'Not Found'
'''
encoding1=get_face_encodings("D:\\WORK_SPACE\\face_recognition\\img1.jpg")
enc=get_face_encodings("D:\\WORK_SPACE\\face_recognition\\bill.jpg")
encoding1.append(enc[0])
print(encoding1)
print(enc)

encoding2=get_face_encodings("D:\\WORK_SPACE\\face_recognition\\bill3.jpg")
print(encoding2[0])

out=compare_face_encodings(encoding1,encoding2[0])
print(out)

match=find_match(encoding1, names, encoding2[0])
print(match)'''




