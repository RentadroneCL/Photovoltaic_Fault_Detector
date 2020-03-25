#! /usr/bin/env python

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

def _main_(args):

    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.8, 0.3

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif input_path[-4:] == '.mp4': # do detection on a video
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               50.0,
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)

                        # show the video with detection bounding boxes
                        if show_window: cv2.imshow('video with bboxes', images[i])

                        # write result to the output video
                        video_writer.write(images[i])
                    images = []

                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()
    else: # do detection on an image or a set of images



        image_paths = []

        if os.path.isdir(input_path):
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        times = []
        images = [cv2.imread(image_path) for image_path in image_paths]
        
        print(images)
        start = time.time()
        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
        boxes = [[box for box in boxes_image if box.get_score() > obj_thresh] for boxes_image in boxes]
        
        print('Elapsed time = {}'.format(time.time() - start))
        times.append(time.time() - start)
        
        boxes_disc = [disconnect(image, boxes_image, z_thresh = 1.8) for image, boxes_image in zip(images, boxes)]
            
        for image, boxes_image in zip(images, boxes_disc):
            
            
            # draw bounding boxes on the image using labels
            I = image.copy()
            draw_boxes(I, boxes_image, config['model']['labels'], obj_thresh)
            plt.figure(figsize = (10,12))
            plt.imshow(I)
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))

        file = open(args.output + '/time.txt','w')
        file.write('Tiempo promedio:' + str(np.mean(times)))
        file.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

    args = argparser.parse_args()
    _main_(args)
