#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################
    valid_ints, labels = parse_voc_annotation(
        config['test']['test_annot_folder'],
        config['test']['test_image_folder'],
        config['test']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)

    valid_generator = BatchGenerator(
        instances           = valid_ints,
        anchors             = config['model']['anchors'],
        labels              = labels,
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],
        shuffle             = True,
        jitter              = 0.0,
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'])

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    total_instances = []
    precisions = []
    print(average_precisions.items())
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              labels[label], 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) == 0:
        print('No test instances found.')
        return

    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
    print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
