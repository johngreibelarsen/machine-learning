# Import libraries necessary for this project
from sklearn.datasets import load_files       
from keras.utils import np_utils
import cv2                
import matplotlib.pyplot as plt                        
import numpy as np
from glob import glob
import random

def main():
    print("\n ----------------------- Now humans --------------------- \n")

    # load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/lfw/*/*"))
    
    # print statistics about the dataset
    print('There are %d total human images.' % len(human_files)) 
               
    human_files_short = human_files[:1000]
    
    frontalface = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # H: 100%
    profileface = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml') # H: 43%
    smile = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml') # H: 100%
    lefteye = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml') # H: 83%
    righteye = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml') # H: 90%
    upperbody = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml') # H: 3%

    human_faces_counted = 0
    for img_path in human_files_short:
        if face_detector(img_path, frontalface):
            human_faces_counted += 1    
        elif face_detector(img_path, profileface):
            human_faces_counted += 1    
        elif face_detector(img_path, smile) and (face_detector(img_path, lefteye) or face_detector(img_path, righteye)):
            human_faces_counted += 1   
        elif face_detector(img_path, upperbody):
            human_faces_counted += 1   
             
    print("What percentage of the first 1000 images in human_files have a detected human face: ", human_faces_counted)


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if (faces == None):
        return false
    return len(faces) > 0


if __name__ == "__main__": main()