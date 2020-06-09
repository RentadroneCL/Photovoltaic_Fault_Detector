#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:55:42 2020

@author: dlsaavedra
"""
import numpy as np

def disconnect(image, boxes, obj_thresh = 0.5, area_min = 400, merge = 0, z_thresh = 1.8):

    new_boxes = []
    for num, box in enumerate(boxes):

        xmin = box.xmin + merge
        xmax = box.xmax - merge
        ymin = box.ymin + merge
        ymax = box.ymax - merge

        if xmin > 0 and ymin > 0 and xmax < image.shape[1] and ymax < image.shape[0] and box.get_score() > obj_thresh:

            area = (ymax - ymin)*(xmax - xmin)
            z_score = np.sum(image[np.int(ymin):np.int(ymax), np.int(xmin):np.int(xmax)]) / area

            if area > area_min:

                box.z_score = z_score
                new_boxes.append(box)
                #boxes_area_score[str(num)] = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'score' : score, 'area' : area}

    mean_score = np.mean([box.z_score for box in new_boxes])
    sd_score  = np.std([box.z_score for box in new_boxes])

    new_boxes = [box for box in new_boxes if (box.z_score - mean_score)/sd_score > z_thresh]

    for box in new_boxes:

        z_score = (box.z_score - mean_score)/sd_score
        box.classes[0] = min((z_score-z_thresh)*0.5/(3-z_thresh)+ 0.5, 1)
        box.score  = -1
    return new_boxes




def disconnect_plot(image, boxes,  obj_thresh = 0.5, area_min = 400, merge = 0,  z_thresh = 1.8):

    new_boxes = []
    for num, box in enumerate(boxes):

        xmin = box.xmin + merge
        xmax = box.xmax - merge
        ymin = box.ymin + merge
        ymax = box.ymax - merge

        if xmin > 0 and ymin > 0 and xmax < image.shape[1] and ymax < image.shape[0] and box.get_score() > obj_thresh:

            area = (ymax - ymin)*(xmax - xmin)
            z_score = np.sum(image[np.int(ymin):np.int(ymax), np.int(xmin):np.int(xmax)]) / area

            if area > area_min:

                box.z_score = z_score
                new_boxes.append(box)
                #boxes_area_score[str(num)] = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'score' : score, 'area' : area}

    mean_score = np.mean([box.z_score for box in new_boxes])
    sd_score  = np.std([box.z_score for box in new_boxes])

    normal_score = ([box.z_score for box in new_boxes] - mean_score)/sd_score
#        plt.figure()
#         _ = plt.hist(normal_score, bins='auto')  # arguments are passed to np.histogram
#        plt.title("Histogram with 'auto' bins")
#        plt.show()
#
#        plt.figure()
#        mean = np.mean([boxes_area_score[i]['area'] for i in boxes_area_score])
#        sd  = np.std([boxes_area_score[i]['area'] for i in boxes_area_score])
#        normal = ([boxes_area_score[i]['area'] for i in boxes_area_score] - mean)/sd
#        _ = plt.hist(normal, bins='auto')  # arguments are passed to np.histogram
#        plt.title("Histogram with 'auto' bins")
#        plt.show()

    new_boxes = [box for box in new_boxes if (box.z_score - mean_score)/sd_score > z_thresh]

    for box in new_boxes:

        z_score = (box.z_score - mean_score)/sd_score
        box.classes[0] = min((z_score-z_thresh)*0.5/(3-z_thresh)+ 0.5, 1)




    colors = plt.cm.brg(np.linspace(0, 1, 21)).tolist()
    plt.figure(figsize=(10,6))
    plt.imshow(I,cmap = 'gray')
    current_axis = plt.gca()

    for box in new_boxes:

        color = colors[2]

        #boxes_area_score[key]['score_norm'] = (boxes_area_score[key]['score'] - mean) / sd
        #z_score = (box.score - mean_score) / sd_score
        #z_score = (boxes_area_score[key]['area'] )

        ### Escribe el z-score
        #if z_score > 1:
        current_axis.text((box.xmin + box.xmax)/2,
                              (box.ymin+ box.ymax)/2,
                               '%.2f' % box.classes[0], size='x-large',
                              color='white', bbox={'facecolor':color, 'alpha':1.0})

    return new_boxes
