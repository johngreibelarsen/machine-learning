# Import libraries necessary for this project
from sklearn.datasets import load_files       
from keras.utils import np_utils
import cv2                
import matplotlib.pyplot as plt                        
import numpy as np
from glob import glob
import random

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    #print(data.keys())
    # data keys are dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
    # print(data['filenames'])
    # numpy arrays containing file paths to images like:
    # 'dogImages/test\\057.Dalmatian\\Dalmatian_04056.jpg'
    # 'dogImages/test\\059.Doberman_pinscher\\Doberman_pinscher_04156.jpg'
    # print(data['target'])
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    # For each image we have a 133 element vector (133 dog categories) with one 1 for the specific dog category the image portraits
    # print(dog_targets.shape)
    return dog_files, dog_targets


def main():
    print('Loading data...')

    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    #valid_files, valid_targets = load_dataset('dogImages/valid')
    #test_files, test_targets = load_dataset('dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    #print('There are %d total dog categories.' % len(dog_names))
    #print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    #print('There are %d training dog images.' % len(train_files))
    #print('There are %d validation dog images.' % len(valid_files))
    #print('There are %d test dog images.'% len(test_files))

    print("\n ----------------------- Now humans --------------------- \n")
    random.seed(8675309)

    # load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/lfw/*/*"))
    random.shuffle(human_files)
    
    # print statistics about the dataset
    print('There are %d total human images.' % len(human_files)) 
       
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    
    # load color (BGR) image
    print("human files {}", human_files[3])
    img = cv2.imread(human_files[3])
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find faces in image
    faces = face_cascade.detectMultiScale(gray)
    print(faces)
    # print number of faces detected in the image
    print('Number of faces detected:', len(faces))
    
    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display the image, along with bounding box
    plt.imshow(cv_rgb)
    plt.show()
    
    human_files_short = human_files[:100]
    dog_files_short = train_files[:100]
    
    human_faces_counted = 0
    
    """    for img_path in human_files_short:
        if face_detector(img_path, face_cascade):
            human_faces_counted += 1    
    print("What percentage of the first 100 images in human_files have a detected human face: {}", human_faces_counted)

    human_faces_counted = 0
    for img_path in dog_files_short:
        if face_detector(img_path, face_cascade):
            human_faces_counted += 1    
    print("What percentage of the first 100 images in dog_files have a detected human face: {}", human_faces_counted)
    
    """    

    #frontal_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    profile_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    human_faces_counted = 0
    for img_path in human_files_short:
        if face_detector(img_path, face_cascade):
            human_faces_counted += 1    
        elif face_detector(img_path, profile_face):
            human_faces_counted += 1  
    print("What percentage of the first 100 images in human_files have a detected human face: {}", human_faces_counted)


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print(faces)
    print(type(faces))
    if (faces == None):
        print('None')
    return len(faces) > 0


if __name__ == "__main__": main()