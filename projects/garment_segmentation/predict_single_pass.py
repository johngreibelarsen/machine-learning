import cv2
import glob
import numpy as np
from sklearn.metrics import f1_score

    
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold

model = params.model_factory()
model.load_weights(filepath = params.weight_no_focus_path)


def generate_mask(img):    
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
       
    img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, axis=0)
    
    out_path = "./normalization/"    
    img_mean = np.load(out_path + "unet_no_focus_mask_1024_meanstd_resize_optimized_mean.npy")
    img_std = np.load(out_path + "unet_no_focus_mask_1024_meanstd_resize_optimized_std.npy")
    
#    out_path = "./input_CP/"    
#    img_mean = np.load(out_path + "original_mean.npy")
#    img_std = np.load(out_path + "original_std.npy")

    img -= img_mean # zero-center
    img /= img_std # normalize

    
    prediction = model.predict(img)
    prediction = np.squeeze(prediction, axis=(0, 3))
    prediction = cv2.resize(prediction, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    return prediction
 

def show_metrics_for_mask(img_name, img, mask):
    predicted_mask = generate_mask(img)
    predicted_mask = predicted_mask > threshold
    mask = mask >= 1
    dice_score = str(f1_score(mask.flatten(), predicted_mask.flatten(), average='binary'))
    print(img_name + ', ' + dice_score)


# Demo showing generation of mask and saving it
if __name__ == '__main__':
#    img_orig_path = "./input/test_double_pass/original/*.*"
#    img_mask_path = "./input/test_double_pass/masks/*.*"
#    img_orig_path = "./input_CP/fold_1_unseen/original/*.*"
#    img_mask_path = "./input_CP/fold_1_unseen/masks/*.*"
    img_orig_path = "./udacity_docs/outliers/original/*.*"
    img_mask_path = "./udacity_docs/outliers/masks/*.*"

    img_path_list = sorted(glob.glob(img_orig_path))
    mask_path_list = sorted(glob.glob(img_mask_path))
    
    print(len(img_path_list))
    print(len(mask_path_list))
    for index in range(0, len(img_path_list)):
        img_orig = cv2.imread(img_path_list[index])
        img_mask = cv2.imread(mask_path_list[index], 0)
        show_metrics_for_mask(img_path_list[index], img_orig, img_mask)
        