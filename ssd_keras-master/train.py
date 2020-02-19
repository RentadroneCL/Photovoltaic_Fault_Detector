"""
Created on Fri May  10 15:10:46 2019

@author: dlsaavedra
"""
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import json
import os
import argparse

K.tensorflow_backend._get_available_gpus()


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    path_imgs_training = config['train']['train_image_folder']
    path_anns_training = config['train']['train_annot_folder']
    path_imgs_val =  config['valid']['valid_image_folder']
    path_anns_val = config['valid']['valid_annot_folder']
    labels = config['model']['labels']
    categories = {}
    #categories = {"Razor": 1, "Gun": 2, "Knife": 3, "Shuriken": 4} #la categoría 0 es la background
    for i in range(len(labels)): categories[labels[i]] = i+1
    print('\nTraining on: \t' + str(categories) + '\n')

    ####################################
    #   Parameters
    ###################################
        #%%
    img_height = config['model']['input'] # Height of the model input images
    img_width = config['model']['input'] # Width of the model input images
    img_channels = 3 # Number of color channels of the model input images
    mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = len(labels) # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    #scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales = scales_pascal
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True

    K.clear_session() # Clear previous models from memory.


    model_path = config['train']['saved_weights_name']
    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.


    if config['model']['backend'] == 'ssd512':
        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
        steps = [8, 16, 32, 64, 100, 200, 300] # The space between two adjacent anchor box center points for each predictor layer.
        offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        scales = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]

    elif config['model']['backend'] == 'ssd7':
        #weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
        scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
        aspect_ratios = [0.5 ,1.0, 2.0] # The list of aspect ratios for the anchor boxes
        two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
        steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
        offsets = None

    if os.path.exists(model_path):
        print("\nLoading pretrained weights.\n")
        # We need to create an SSDLoss object in order to pass that to the model loader.
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        K.clear_session() # Clear previous models from memory.
        model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})


    else:
        ####################################
        #   Build the Keras model.
        ###################################

        if config['model']['backend'] == 'ssd300':
            #weights_path = 'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
            from models.keras_ssd300 import ssd_300 as ssd

            model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)


        elif config['model']['backend'] == 'ssd512':
            #weights_path = 'VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'
            from models.keras_ssd512 import ssd_512 as ssd

            # 2: Load some weights into the model.
            model = ssd(image_size=(img_height, img_width, img_channels),
                            n_classes=n_classes,
                            mode='training',
                            l2_regularization=0.0005,
                            scales=scales,
                            aspect_ratios_per_layer=aspect_ratios,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=clip_boxes,
                            variances=variances,
                            normalize_coords=normalize_coords,
                            swap_channels=swap_channels)

        elif config['model']['backend'] == 'ssd7':
            #weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
            from models.keras_ssd7 import build_model as ssd
            scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
            aspect_ratios = [0.5 ,1.0, 2.0] # The list of aspect ratios for the anchor boxes
            two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
            steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
            offsets = None
            model = ssd(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
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
                    subtract_mean=None,
                    divide_by_stddev=None)

        else :
            print('Wrong Backend')



        print('OK create model')
         #sgd = SGD(lr=config['train']['learning_rate'], momentum=0.9, decay=0.0, nesterov=False)

        # TODO: Set the path to the weights you want to load. only for ssd300 or ssd512

        weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
        print("\nLoading pretrained weights VGG.\n")
        model.load_weights(weights_path, by_name=True)

        # 3: Instantiate an optimizer and the SSD loss function and compile the model.
        #    If you want to follow the original Caffe implementation, use the preset SGD
        #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.


        #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        optimizer = Adam(lr=config['train']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=optimizer, loss=ssd_loss.compute_loss)

        model.summary()

    #####################################################################
    #  Instantiate two `DataGenerator` objects: One for training, one for validation.
    ######################################################################
    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.



    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    classes = ['background'] + labels

    train_dataset.parse_xml(images_dirs= [config['train']['train_image_folder']],
                            image_set_filenames=[config['train']['train_image_set_filename']],
                            annotations_dirs=[config['train']['train_annot_folder']],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs= [config['valid']['valid_image_folder']],
                            image_set_filenames=[config['valid']['valid_image_set_filename']],
                            annotations_dirs=[config['valid']['valid_annot_folder']],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    #########################
    # 3: Set the batch size.
    #########################
    batch_size = config['train']['batch_size'] # Change the batch size if you like, or if you run into GPU memory issues.

    ##########################
    # 4: Set the image transformations for pre-processing and data augmentation options.
    ##########################
    # For the training generator:


    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    ######################################3
    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
    #########################################
    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    if config['model']['backend'] == 'ssd512':
        predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                           model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

        ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                            img_width=img_width,
                                            n_classes=n_classes,
                                            predictor_sizes=predictor_sizes,
                                            scales=scales,
                                            aspect_ratios_per_layer=aspect_ratios,
                                            two_boxes_for_ar1=two_boxes_for_ar1,
                                            steps=steps,
                                            offsets=offsets,
                                            clip_boxes=clip_boxes,
                                            variances=variances,
                                            matching_type='multi',
                                            pos_iou_threshold=0.5,
                                            neg_iou_limit=0.5,
                                            normalize_coords=normalize_coords)

    elif config['model']['backend'] == 'ssd300':
        predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                           model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
        ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                            img_width=img_width,
                                            n_classes=n_classes,
                                            predictor_sizes=predictor_sizes,
                                            scales=scales,
                                            aspect_ratios_per_layer=aspect_ratios,
                                            two_boxes_for_ar1=two_boxes_for_ar1,
                                            steps=steps,
                                            offsets=offsets,
                                            clip_boxes=clip_boxes,
                                            variances=variances,
                                            matching_type='multi',
                                            pos_iou_threshold=0.5,
                                            neg_iou_limit=0.5,
                                            normalize_coords=normalize_coords)

    elif config['model']['backend'] == 'ssd7':
        predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                           model.get_layer('classes5').output_shape[1:3],
                           model.get_layer('classes6').output_shape[1:3],
                           model.get_layer('classes7').output_shape[1:3]]
        ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)



    #######################
    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
    #######################

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=  [SSDDataAugmentation(img_height=img_height,img_width=img_width)],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size   = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))



    ##########################
    # Define model callbacks.
    #########################

    # TODO: Set the filepath under which you want to save the model.
    model_checkpoint = ModelCheckpoint(filepath= config['train']['saved_weights_name'],
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    #model_checkpoint.best =

    csv_logger = CSVLogger(filename='log.csv',
                           separator=',',
                           append=True)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)

    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                 csv_logger,
                 learning_rate_scheduler,
                 terminate_on_nan]



    #print(model.summary())
    batch_images, batch_labels = next(train_generator)

#    i = 0 # Which batch item to look at
#
#    print("Image:", batch_filenames[i])
#    print()
#    print("Ground truth boxes:\n")
#    print(batch_labels[i])




    initial_epoch   = 0
    final_epoch     = config['train']['nb_epochs']
    #final_epoch     = 20
    steps_per_epoch = 500

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size/batch_size),
                                  initial_epoch=initial_epoch,
                                  verbose = 1 if config['train']['debug'] else 2)




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
