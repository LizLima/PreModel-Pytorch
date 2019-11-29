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
        p = "Datasets/shape_predictor_68_face_landmarks.dat"
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

    def getShapeSquare( self, pointX, pointY, width, height):

            maxDist = max(width, height)

            # middleX
            midX = pointX + round(width/2)
            midY = pointY + round(height/2)

            pointX = max(midX - round(maxDist/2), 0)
            pointY = max(midY - round(maxDist/2), 0)

            return pointX, pointY, maxDist, maxDist
        
    def getShapeAspect(self,pointX, pointY, w_shape, h_shape, width, height):
        
        factor = 0
        aspect = w_shape/h_shape
        ideal_aspect  = width/height
        new_width = 0
        new_height = 0
        if aspect < ideal_aspect:
            # Then crop the left and right edges:
            new_width = int(ideal_aspect * h_shape)
            offset = (w_shape - new_width) / 2
            new_width = int(w_shape - offset + factor)
            new_height = int(h_shape + factor)
        else:
            # ... crop the top and bottom:
            new_height = int(w_shape / ideal_aspect)
            offset = (h_shape - new_height) / 2
            new_width = int(w_shape + factor)
            new_height = int(h_shape - offset + factor)

        # Middle of original shape
        midX = pointX + round(w_shape/2)
        midY = pointY + round(h_shape/2)

        pointX = max(midX - round(new_width/2), 0)
        pointY = max(midY - round(new_height/2), 0)

        return pointX, pointY, new_width, new_height
    def getPatches(self, imageCV):

        #Detect face in the image
        rects = self.detector(imageCV, 0)
        # Patches initialization
        crop_left_eye =  np.zeros((40,40,3), np.uint8) #torch.zeros(3,40,40) #  np.zeros((height,width,3), np.uint8)
        crop_right_eye = np.zeros((40,40,3), np.uint8)
        crop_nose = np.zeros((32,40,3), np.uint8) #torch.zeros(3,32,40)
        crop_mouth = np.zeros((48,32,3), np.uint8) #torch.zeros(3,32,48)
        
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
            pointX, pointY, width, height  = self.getShapeSquare( pointX, pointY, width, height)
            crop_left_eye = imageCV[pointY : pointY + height , pointX : pointX + width ]
            
            # Right eye
            pointX, pointY, width, height  = cv2.boundingRect(shape_right_eye)
            pointX, pointY, width, height  = self.getShapeSquare( pointX, pointY, width, height)
            crop_right_eye = imageCV[pointY : pointY + height , pointX : pointX + width ]

            # Nose
            pointX, pointY, width, height  = cv2.boundingRect(shape_nose)
            pointX, pointY, width, height  = self.getShapeAspect(pointX, pointY, width, height, 32, 40)
            crop_nose = imageCV[pointY : pointY + height , pointX : pointX + width ]

            # Mouth
            pointX, pointY, width, height  = cv2.boundingRect(shape_mouth)
            pointX, pointY, width, height  = self.getShapeAspect(pointX, pointY, width, height, 48, 32)
            crop_mouth = imageCV[pointY : pointY + height , pointX : pointX + width ]
           
        return crop_left_eye, crop_right_eye, crop_nose, crop_mouth

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



# Main
#  h,w, c= crop_mouth.shape
#                 if w == 0:
#                     print("error")

# img = cv2.imread('01.jpg',1)
# land = Landmark(3)

# p1, p2, p3, p4 = land.getPatches(img)

# print(p1.shape)
# print(p2.shape)
# print(p3.shape)
# print(p4.shape)

# cv2.imshow("image", img)
# cv2.imshow("p1", p1)
# cv2.imshow("p2", p2)
# cv2.imshow("p3", p3)
# cv2.imshow("p4", p4)
# cv2.waitKey(30000)
