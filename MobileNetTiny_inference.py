
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from Architecture.MobilenetV2 import build_model
from loss_function.keras_ssd_loss import SSDLoss
from layers.keras_layer_AnchorBoxes import AnchorBoxes
from layers.keras_layer_DecodeDetections import DecodeDetections
from layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from encoder_decoder.ssd_input_encoder import SSDInputEncoder
from encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import cv2
import time

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']


img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='inference_fast',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Optional: Load some weights

model.load_weights('new_mobv2_epoch-45_val_loss-2.0145.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


def pre_pro_img(frame):
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    orig_images.append(frame)
    #img = image.load_img(frame, target_size=(img_height, img_width))
    img = cv2.resize(orig_images[0], (480, 300))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)
    
    return orig_images, input_images


def predict(orig_images, input_images):
    y_pred = model.predict(input_images)
    
    confidence_threshold = 0.6
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = int(box[-4] * orig_images[0].shape[1] / img_width)
        ymin = int(box[-3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[-2] * orig_images[0].shape[1] / img_width)
        ymax = int(box[-1] * orig_images[0].shape[0] / img_height)
        color = COLORS[int(box[0]) % 3]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), color, 2)
        
        text_top = (xmin, ymin-10)
        text_bot = (xmin + 80, ymin + 5)
        text_pos = (xmin + 5, ymin)
        cv2.rectangle(orig_images[0], text_top, text_bot, color, -1)
        cv2.putText(orig_images[0], label, text_pos, FONT, 0.35, (0,0,0), 1)
    return orig_images[0]


cap = cv2.VideoCapture('examples/sample3.mp4')

while(True):
    ret, frame = cap.read()
    start_time = time.time()
    orig_images, input_images = pre_pro_img(frame)
    pred_frame = predict(orig_images, input_images)
    print(time.time() - start_time)
    cv2.imshow('frame', pred_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


