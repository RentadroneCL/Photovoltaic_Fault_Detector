#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:27:19 2020

@author: dlsaavedra
"""

#! /usr/bin/env python

import argparse
import os
import numpy as np
import errno
import flirimageextractor
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(
    description = 'Change flirt image to jpg image')



argparser.add_argument(
    '-i',
    '--input',
    help='path to an folder of image')


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
                
def _main_(args):
    
    input_path   = args.input

    
    

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    
    for f in files:
        flir = flirimageextractor.FlirImageExtractor()
        print(f)
        try:
            flir.process_image(f)
            I = flirimageextractor.FlirImageExtractor.get_thermal_np(flir)
        except:
            I = plt.imread(f)
        #flir.save_images()
        #flir.plot()
        
        
        
        #img = img.astype(np.int8) 
        W = np.where(np.isnan(I))
        if np.shape(W)[1] > 0:

            #xmax = np.max(np.amax(W,axis=0))
            ymax = np.max(np.amin(W,axis=1))
            img = I[:ymax,:]
        else:
            img = I
        
        list_string = f.split('/')
        list_string[-3]+= '_jpg'
        f_aux = '/'.join(list_string)
        
        mkdir(f_aux)
        plt.imsave(f_aux, img, cmap = 'gray')
        


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)