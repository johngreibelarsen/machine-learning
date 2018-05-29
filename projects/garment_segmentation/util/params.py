from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

"""
    Our config file for training and predicting our U-net model
"""

# Image sizes
orig_width = 1152 # original width of image
orig_height = 1728 # original height of image


# Model parameters
model_lr=5e-3 # The default learning rate that the a selected model is born with
input_size = 1024 # The image input size to the model: 128, 256, 512 or 1024
max_epochs = 100 # Max epochs to use for training
batch_size = 20 # Batch size to use for reading and processing images
model_factory = get_unet_1024 # The model choosen


# Learning rate parameters (for use with cyclic_learning_rate.py)
plateau_steps = 1 # The initial no. of epocs were the LR will be constant at (min_lr + max_lr)/2
step_size = 11 # 1/2 the wave length of the oscillating LR function
min_lr = 7e-7 # Min lR
max_lr = 7e-4 # Max LR


# Mask cut-out parameters
threshold = 0.3 # the probability threshold above which a pixel is decided as part of the image mask


# Model weights to use for predictions
weight_no_focus_path = './weights/unet_no_mask_1024_meanstd_downsize_optimized.hdf5'
#weight_no_focus_path = './weights/unet_no_mask_256_preprocessing_downsize_optimized_color_V2.hdf5'
#weight_no_focus_path = './weights/metail_machine/weights/unet_no_mask_1024_meanstd_all_layers_fold_2.hdf5'
#weight_no_focus_path = './weights/unet_1024_no_mask.hdf5'
#weight_no_focus_path = './weights/unet_1024_with_no_mask_1304_basic_preprocessing_advanced_color_900.hdf5'
#weight_with_focus_path = './weights/unet_1024_with_focus_mask_2703_adadelta_1.hdf5' # Image through dilated mask
#weight_with_focus_path = './weights/unet_1024_with_focus_mask_weighted_2903_1.hdf5' # Image through dilated mask
#weight_with_focus_path = './weights/unet_1024_with_focus_SGD_2703.hdf5' # Image through dilated mask
#weight_with_focus_path = './weights/unet_1024_with_focus_Adam_2703.hdf5' # Image through dilated mask
#weight_with_focus_path = './weights/unet_1024_with_focus_Adadelta_2703.hdf5' # Image through dilated mask
weight_with_focus_path = './weights/unet_1024_with_mask.hdf5' # Image through dilated mask
