import cv2
import glob
import numpy as np
import skimage
from sklearn.metrics import f1_score
    
from util.remove_background import focus_garment_image
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold

model1 = params.model_factory()
model1.load_weights(filepath = params.weight_no_focus_path)

model2 = params.model_factory()
model2.load_weights(filepath = params.weight_with_focus_path)


def generate_initial_mask(img):    
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, axis=0)

    prediction = model1.predict(img)
    prediction = np.squeeze(prediction, axis=(0, 3))
    prediction = cv2.resize(prediction, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    prediction = prediction > 0.67 #0.55 #0.5 #0.45
    return prediction
 

def generate_final_mask(img, mask):  
    mask = np.expand_dims(mask, axis=2)
    mask = skimage.img_as_ubyte(mask, force_copy=False)
    
    focused_img = focus_garment_image(img, mask)
    focused_img = cv2.resize(focused_img, (input_size, input_size),  interpolation=cv2.INTER_AREA)
    focused_img = np.array(focused_img, np.float32) / 255
    focused_img = np.expand_dims(focused_img, axis=0)

    prediction = model2.predict(focused_img)
    prediction = np.squeeze(prediction, axis=(0))
    prediction = cv2.resize(prediction, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    prediction = np.expand_dims(prediction, axis=2)
    prediction = prediction > 0.4 #0.35 #0.3 #0.25
    return prediction


def generate_mask(img):
    initial_mask = generate_initial_mask(img)
    prediction = generate_final_mask(img, initial_mask)
    return prediction

def show_metrics_for_masks(img_name, img_orig, img_mask):
    img_mask = img_mask >= 1

    initial_mask = generate_initial_mask(img_orig)
    single_dice = str(f1_score(img_mask.flatten(), initial_mask.flatten(), average='binary'))

    final_mask = generate_final_mask(img_orig, initial_mask)
    final_dice = str(f1_score(img_mask.flatten(), final_mask.flatten(), average='binary'))

    print(img_name + ', ' + single_dice + ', ' + final_dice)

# Demo showing generation of mask and saving it
if __name__ == '__main__':
    img_orig_path = "./input/test_double_pass/original/*.*"
    img_mask_path = "./input/test_double_pass/masks/*.*"
    img_path_list = sorted(glob.glob(img_orig_path))
    mask_path_list = sorted(glob.glob(img_mask_path))
    
    for index in range(0, len(img_path_list)):
        img_orig = cv2.imread(img_path_list[index])
        img_mask = cv2.imread(mask_path_list[index], 0)
        show_metrics_for_masks(img_path_list[index], img_orig, img_mask)
        