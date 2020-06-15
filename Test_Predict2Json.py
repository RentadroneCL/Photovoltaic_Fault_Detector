#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:22:54 2020

@author: dlsaavedra
"""

import time
import os
import argparse
import json
import cv2
import sys
sys.path += [os.path.abspath('keras-yolo3-master')]

from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

#%%


def predict(infer_model_1, infer_model_2, config_1, config_2, images_paths):

    images = []

    for image_path in images_paths:
        image = cv2.imread(image_path)
        images.append(image)

    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################

    labels_1 = config_1['model']['labels']
    labels_2 = config_2['model']['labels']


    boxes_p_1 = get_yolo_boxes(infer_model_1, images, net_h, net_w, config_1['model']['anchors'], obj_thresh, nms_thresh)
    boxes_p_2 = get_yolo_boxes(infer_model_2, images, net_h, net_w, config_2['model']['anchors'], obj_thresh, nms_thresh)

    list_dict = []
    for i in range(len(images)):
        dict_boxes = { 'name_file' : images_paths[i],
                   'objects' : []}
        for boxes in boxes_p_1[i]:
            dict_boxes['objects'].append({
                    'class': labels_1[boxes.label],
                    'score': boxes.score,
                    'xmax':  boxes.xmax,
                    'xmin':  boxes.xmin,
                    'ymax':  boxes.ymax,
                    'ymin':  boxes.ymax
                    })

        for boxes in boxes_p_2[i]:
            dict_boxes['objects'].append({
                    'class': labels_2[boxes.label],
                    'score': boxes.score,
                    'xmax':  boxes.xmax,
                    'xmin':  boxes.xmin,
                    'ymax':  boxes.ymax,
                    'ymin':  boxes.ymax
                    })

        list_dict.append(dict_boxes.copy())

    return list_dict



config_model_1 = 'config_full_yolo_fault_1_infer.json'
config_model_2 = 'config_full_yolo_fault_4_infer.json'
input_path = 'fault_jpg/'


with open(config_model_1) as config_buffer:
        config_1 = json.load(config_buffer)
with open(config_model_2) as config_buffer:
    config_2 = json.load(config_buffer)

os.environ['CUDA_VISIBLE_DEVICES'] = config_1['train']['gpus']
infer_model_1 = load_model(config_1['train']['saved_weights_name'])
infer_model_2 = load_model(config_2['train']['saved_weights_name'])

images_paths = []

if os.path.isdir(input_path):
    for inp_file in os.listdir(input_path):
        images_paths += [input_path + inp_file]
else:
    images_paths += [input_path]

images_paths = [inp_file for inp_file in images_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

L = predict(infer_model_1, infer_model_2, config_1, config_2, images_paths)

js = json.dumps(L)
