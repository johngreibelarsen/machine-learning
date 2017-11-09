# Import libraries necessary for this project

import numpy as np
from sklearn.datasets import load_files  
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import ImageFile                            
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from glob import glob
import random

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(ResNet50_model, img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(ResNet50_model, img_path):
    prediction = ResNet50_predict_labels(ResNet50_model, img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def main():
    
    print("START dog detector Main")
    
    random.seed(8675309)

    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')

    train_files = train_files[:1000] 
    train_targets = train_targets[:1000]

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    dog_files_short = train_files[:100]
        
    dog_faces_counted = 0
    for img_path in dog_files_short:
        if dog_detector(ResNet50_model, img_path):
            dog_faces_counted += 1    
    print("What percentage of the first 1000 images in dog_files have a detected dog face: {}%".format(dog_faces_counted/10))
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True  

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255

    ### TODO: Define your architecture.
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=4, strides=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(GlobalAveragePooling2D())   
    model.add(Dense(133, activation='softmax'))

    model.summary()
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    ### TODO: specify the number of epochs that you would like to use to train the model.    
    epochs = 5
    
    ### Do NOT modify the code below this line.
    
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                                   verbose=1, save_best_only=True)
    
    model.fit(train_tensors, train_targets, 
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    model.load_weights('saved_models/weights.best.from_scratch.hdf5')
    
    # get index of predicted dog breed for each image in test set
    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    print("dog_breed_predictions: {}%".format(dog_breed_predictions))
    print('Test accuracy: %.4f%%' % test_accuracy)
    



    print("END dog detector Main")

if __name__ == "__main__": main()