import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt  
import skimage
from tqdm import tqdm
from sklearn.metrics import f1_score
    
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold

model = params.model_factory()
model.load_weights(filepath = params.weight_no_focus_path)


def load_images_preprocessed(img_path):    
    if not img_path.endswith("/*.*"): img_path = img_path + "/*.*"
    img_path = sorted(glob.glob(img_path))
    img_list = []
    for file in tqdm(img_path):
        img = cv2.imread(file)
        img = cv2.resize(img, (input_size, input_size))
        img_list.append(img)
    return np.array(img_list, np.float32) / 255   


def generate_masks(img_orig_paths, img_mask_paths):    
    orig_img_list = load_images_preprocessed(img_orig_paths)
    mask_img_list = load_images_preprocessed(img_mask_paths)
    
    prediction_list = model.predict_on_batch(orig_img_list)
    prediction_list = np.squeeze(prediction_list, axis=3) # remove channel ex. (4, 1024, 1024, 1) --> (4, 1024, 1024)
    
    for index in tqdm(range(len(prediction_list))):
        original = cv2.resize(orig_img_list[index], (orig_width, orig_height))
        prediction = cv2.resize(prediction_list[index], (orig_width, orig_height))
        mask_ground_truth = cv2.resize(mask_img_list[index], (orig_width, orig_height))       
    
        show_images(original, prediction, mask_ground_truth)


def show_images(original, prediction, mask_ground_truth):
    
    mask_predicted = prediction > threshold 
    mask_ground_truth = cv2.cvtColor(mask_ground_truth, cv2.COLOR_BGR2GRAY)
    mask_ground_truth = mask_ground_truth > threshold
    
    print("\n***********************************")
    print("Dice f1: " + str(f1_score(mask_ground_truth.flatten(), mask_predicted.flatten(), average='binary')))
    print("***********************************")
       
    fig = plt.figure(figsize=(32, 32))
    
    imgplt = fig.add_subplot(1, 4, 1)
    imgplt.set_title("Original")
    plt.imshow(original)
    
    imgplt = fig.add_subplot(1, 4, 2)
    imgplt.set_title("Our Prediction")
    plt.imshow(prediction)
 
    imgplt = fig.add_subplot(1, 4, 3)
    imgplt.set_title("ISI cut-out")
    plt.imshow(mask_ground_truth)

    imgplt = fig.add_subplot(1, 4, 4)
    imgplt.set_title("our auto cut-out")
    
    original = skimage.img_as_ubyte(original, force_copy=False)
    prediction = skimage.img_as_ubyte(prediction, force_copy=False)
    our_cutout = cv2.bitwise_and(original, original, mask=prediction)  

    plt.imshow(our_cutout)
    
    plt.show()


# Demo showing original image next to our prediction masks followed by DCIL/ISI cut-out 
# mask and finally the effect of applying our mask to the original image
if __name__ == '__main__':
    img_orig_paths = "./input/test_garment_masks/original"
    img_mask_paths = "./input/test_garment_masks/masks"
    generate_masks(img_orig_paths, img_mask_paths)
    