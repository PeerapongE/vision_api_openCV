# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:07:30 2019

@author: PeerapongE
"""
# face bound
# (167,112),(443,112),(443,433),(167,433)
# tutorial
# https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/

#import numpy as np
import cv2
import os
from google.cloud import vision
import io
import pandas as pd
import random



def detect_faces(path):
    """Detects faces in an image."""
    #from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    # value are 0,1,2,3,4,5 respectively
    
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('---- Faces: ----')
    print('Number of faces found = %d'%len(faces))
    
    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))
    return faces


def get_face_property(faces):
    # function to get face property as boundary (list) and the highest chance emotion

    face_bounary_collector = []
    face_emotion_collector = []
    
    for face in faces:
    
        # get boundary for rectangle creation
        face_boundary = [(face.bounding_poly.vertices[0].x , face.bounding_poly.vertices[0].y) , (face.bounding_poly.vertices[2].x , face.bounding_poly.vertices[2].y)]
        # get face emotion as dict
        face_emotion_dict  = {'Joyful':face.joy_likelihood,
                              'Sorrow':face.sorrow_likelihood,
                              'Anger':face.anger_likelihood,
                              'Surprise':face.surprise_likelihood}
        # create series adn sort descending --> select the highest chance emotion to show
        face_emotion_best = pd.Series(face_emotion_dict).sort_values(ascending = False).index[0] # get the best emotion
    
        face_bounary_collector.append(face_boundary)
        face_emotion_collector.append(face_emotion_best)
        
    return (face_bounary_collector, face_emotion_collector)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP-Vision-API-305a4d97586d.json" # set environment path
# register with google service: https://cloud.google.com/vision/docs/quickstart-client-libraries


#path = 'Peet_test.jpg'
#path = 'two_people.jpg'
#path = '3_people.jpg'
#path = 'PM_Election.jpeg'
path = 'five_people_election.jpg' # put 

faces = detect_faces(path)

img = cv2.imread(path,cv2.IMREAD_COLOR)
(face_bounary_collector, face_emotion_collector) = get_face_property(faces)

# draw picture
for boundary, emotion  in zip(face_bounary_collector, face_emotion_collector):

    #font = cv2.FONT_HERSHEY_SIMPLEX
    color_red = random.randint(0, 255) # random color for red  
    color_green = random.randint(0, 255) # random color for green  
    color_blue = random.randint(0, 255) # random color for blue
    
    cv2.rectangle(img,boundary[0],boundary[1],(color_red,color_green,color_blue),3) # add regangle
    
    
    #cv2.putText(img, emotion, (boundary[0][0],boundary[0][1]-10), font, 1, (color_red,color_green,color_blue), 2, cv2.LINE_AA) # add text emotion
    
    cv2.putText(img  = img,
                text = emotion, 
                org  = (boundary[0][0],boundary[0][1]-10),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1,
                color = (color_red,color_green,color_blue),
                thickness = 2,
                lineType = cv2.LINE_AA) # add text emotion
    
    



cv2.imshow('image_show',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# saveoutput
cv2.imwrite('output.png',img)