#Reference
# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672

from imutils import face_utils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
class Landmark:
    def __init__(self, factor):
        self.factor = factor
        p = "Datasets/FaceLadmark/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor( p)

        # Left eye
        self.ini_left_eye   = 36 
        self.last_left_eye  = 42

        # Right eye
        self.ini_right_eye  = 42
        self.last_right_eye = 48

        # Nose
        self.ini_nose       = 27
        self.last_nose      = 36

        # mouth
        self.ini_mouth      = 48
        self.last_mouth     = 68

    def getPoint(self, imageCV):

        #Detect face in the image
        rects = self.detector(imageCV, 0)
        point_leye = None
        point_reye = None
        point_nose = None
        point_mouth = None
        
        # Only one face -> then without for -->  for (i, rect) in enumerate(rects):
        #for (i, rect) in enumerate(rects):
        if len(rects) > 0:
            shape = self.predictor(imageCV, rects[0]) # rect
            
            # array 68 points
            shape = face_utils.shape_to_np(shape)
            # Get point according to landmark.jpeg
            shape_left_eye  = shape[self.ini_left_eye : self.last_left_eye]
            shape_right_eye = shape[self.ini_right_eye : self.last_right_eye]
            shape_nose      = shape[self.ini_nose : self.last_nose]
            shape_mouth     = shape[self.ini_mouth : self.last_mouth]
            
            # Left eye
            pointX, pointY, width, height  = cv2.boundingRect(shape_left_eye)
            midX = pointX + round(width/2)
            midY = pointY + round(height/2)
            point_leye = (midX, midY)
            # Right eye
            pointX, pointY, width, height  = cv2.boundingRect(shape_right_eye)
            midX = pointX + round(width/2)
            midY = pointY + round(height/2)
            point_reye = (midX, midY)
            # Nose
            pointX, pointY, width, height  = cv2.boundingRect(shape_nose)
            midX = pointX + round(width/2)
            midY = pointY + round(height/2)
            point_nose = (midX, midY)
            # Mouth
            pointX, pointY, width, height  = cv2.boundingRect(shape_mouth)
            midX = pointX + round(width/2)
            midY = pointY + round(height/2)
            point_mouth = (midX, midY)
        return [point_leye, point_reye, point_nose, point_mouth]

    def getPointFace(self, imageCV):

        #Detect face in the image
        rects = self.detector(imageCV, 0)
        listPoint = []
        # Only one face -> then without for -->  for (i, rect) in enumerate(rects):
        #for (i, rect) in enumerate(rects):
        if len(rects) > 0:
            shape = self.predictor(imageCV, rects[0]) # rect
            
            # array 68 points
            shape = face_utils.shape_to_np(shape)
            # Left eye
            midX = int((shape[self.ini_left_eye][0] + shape[self.ini_left_eye + 3][0])/2)
            midY = int((shape[self.ini_left_eye][1] + shape[self.ini_left_eye + 3][1])/2)
            point_leye = (midX, midY)
            # Right eye
            midX = int((shape[self.ini_right_eye][0] + shape[self.ini_right_eye + 3][0])/2)
            midY = int((shape[self.ini_right_eye][1] + shape[self.ini_right_eye + 3][1])/2)
            point_reye = (midX, midY)
            # Nose
            point_nose = (shape[self.ini_nose + 3][0], shape[self.ini_nose + 3][1])
            # Mouth
            point_mouth_l = (shape[self.ini_mouth][0], shape[self.ini_mouth][1])
            point_mouth_r = (shape[self.ini_mouth + 6][0], shape[self.ini_mouth + 6][1])
            
            listPoint.append(point_leye)
            listPoint.append(point_reye)
            listPoint.append(point_nose)
            listPoint.append(point_mouth_l)
            listPoint.append(point_mouth_r)

        return listPoint


    def getAllPointFace(self, imageCV):
        #Detect face in the image
        rects = self.detector(imageCV, 0)
        listPoint = []
        # Only one face -> then without for -->  for (i, rect) in enumerate(rects):
        #for (i, rect) in enumerate(rects):
        if len(rects) > 0:
            shape = self.predictor(imageCV, rects[0]) # rect
            
            # array 68 points
            shape = face_utils.shape_to_np(shape)
            listPoint = shape.tolist()
        return listPoint
# Main
