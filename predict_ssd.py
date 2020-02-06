from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import json
import argparse
import os
import time
import sys
sys.path += [os.path.abspath('ssd_keras-master')]

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise



def _main(args=None):
    # parse arguments
    config_path = args.conf
    input_path = args.input_path
    output_path = args.output_path
    
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    makedirs(args.output_path)
    ###############################
    #   Parse the annotations
    ###############################
    score_threshold = 0.5
    labels = config['model']['labels']
    categories = {}
    #categories = {"Razor": 1, "Gun": 2, "Knife": 3, "Shuriken": 4} #la categoría 0 es la background
    for i in range(len(labels)): categories[labels[i]] = i+1
    print('\nTraining on: \t' + str(categories) + '\n')

    img_height = config['model']['input'] # Height of the model input images
    img_width = config['model']['input'] # Width of the model input images
    img_channels = 3 # Number of color channels of the model input images
    n_classes = len(labels) # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    classes = ['background'] + labels

    model_mode = 'training'
    # TODO: Set the path to the `.h5` file of the model to be loaded.
    model_path = config['train']['saved_weights_name']

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session() # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})


   

    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    times = []


    for img_path in image_paths:
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.
        print(img_path)

        # preprocess image for network
        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)
        # process image
        start = time.time()
        y_pred = model.predict(input_images)
        y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=score_threshold,
                                   iou_threshold=score_threshold,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=img_height,
                                   img_width=img_width)
        
        
        print("processing time: ", time.time() - start)
        times.append(time.time() - start)
        # correct for image scale

        # visualize detections
        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.figure(figsize=(20,12))
        plt.imshow(orig_images[0],cmap = 'gray')

        current_axis = plt.gca()
        #print(y_pred)
        for box in y_pred_decoded[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.

            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height

            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        #plt.figure(figsize=(15, 15))
        #plt.axis('off')
        save_path = output_path + img_path.split('/')[-1]
        plt.savefig(save_path)
        plt.close()

    file = open(output_path + 'time.txt','w')

    file.write('Tiempo promedio:' + str(np.mean(times)))

    file.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input_path',       help='folder input.', type=str)
    argparser.add_argument('-o', '--output_path',       help='folder output.', default='ouput/', type=str)
    argparser.add_argument('--score_threshold',       help='score threshold detection.', default=0.5, type=float)
    args = argparser.parse_args()
    _main(args)
