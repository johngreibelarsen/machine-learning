import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt  
    
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold

model = params.model_factory()
model.load_weights(filepath = params.weight_no_focus_path)



def generate_mask(img):    
    print(type(img))
    #print(img)
    img = cv2.resize(img, (input_size, input_size))
    img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    #prediction = cv2.resize(prediction, (orig_width, orig_height))
    print(img.shape)
    print("version 1: " + str(prediction.shape))
    
    prediction = np.squeeze(prediction, axis=3)
    prediction = np.squeeze(prediction, axis=0)

    print("version 2: " + str(prediction.shape))

    prediction = cv2.resize(prediction, (orig_width, orig_height))
    
    print("version 3: " + str(prediction.shape))        
    fig = plt.figure(figsize=(32, 32))
    
    imgplt = fig.add_subplot(1, 2, 1)
    imgplt.set_title("Prediction")
    plt.imshow(prediction)
    
    mask_predicted = prediction > threshold 
    
    imgplt = fig.add_subplot(1, 2, 2)
    imgplt.set_title("Our Mask")
    plt.imshow(mask_predicted)
 


# Demo showing original image next to our prediction masks followed by DCIL/ISI cut-out 
# mask and finally the effect of applying our mask to the original image
if __name__ == '__main__':
    img_orig_paths = "./input/test_garment_masks/original/*.*"
    img_path = sorted(glob.glob(img_orig_paths))
    for file in img_path:
        img_read = cv2.imread(file)
        generate_mask(img_read)
        
        