from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
#from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt
import argparse
import json

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    path_imgs_test = config['test']['test_image_folder']
    path_anns_test = config['test']['test_annot_folder']
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

    test_dataset = DataGenerator()
    test_dataset.parse_xml(images_dirs= [config['test']['test_image_folder']],
                           image_set_filenames=[config['test']['test_image_set_filename']],
                           annotations_dirs=[config['test']['test_annot_folder']],
                           classes=classes,
                           include_classes='all',
                           exclude_truncated=False,
                           exclude_difficult=False,
                           ret=False)
    evaluator = Evaluator(model=model,
                          n_classes=n_classes,
                          data_generator=test_dataset,
                          model_mode=model_mode)

    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=4,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)

    mean_average_precision, average_precisions, precisions, recalls = results

    total_instances = []
    precisions = []
    for i in range(1, len(average_precisions)):
        print('{:.0f} instances of class'.format(len(recalls[i])),
              classes[i], 'with average precision: {:.4f}'.format(average_precisions[i]))
        total_instances.append(len(recalls[i]))
        precisions.append(average_precisions[i])

    if sum(total_instances) == 0:
        print('No test instances found.')
        return

    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
    print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

    for i in range(1, len(average_precisions)):
        print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    print()
    print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))






if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
