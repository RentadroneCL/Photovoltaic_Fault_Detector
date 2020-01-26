#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:12:34 2020

@author: dlsaavedra
"""




import argparse
import os
import numpy as np
import errno
import flirimageextractor
import matplotlib.pyplot as plt
import pandas
import matplotlib.patches as patches
import xml.etree.cElementTree as ET


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                

  


argparser = argparse.ArgumentParser(
    description = 'Data flirt excel to train estructure data')



argparser.add_argument(
    '-i',
    '--input',
    help='path data excel')
argparser.add_argument(
    '-T',
    '--input_thermal',
    help='path thermal images')
    # Example 'Thermal/'
argparser.add_argument(
    '-o',
    '--output',
    help='folder save Train data')
    #Examplo 'Train_B/'


              
def _main_(args):
    
    input_path   = args.input
    output_path = args.output
    thermal_path = args.input_thermal
    
    mkdir(output_path)
    mkdir(output_path + 'images/')
    mkdir(output_path + 'anns/')
    
    Excel = pandas.read_excel(input_path, sheet_name= 'Lista_Archivos_Fotos', header= 1)
    
    for index_path in  range(len(Excel.Archivo)):
        
        if not pandas.notna(Excel.Archivo[index_path]):
            continue
        
        path_Flir = Excel.loc[index_path]['Archivo']
        cod_falla = int(Excel.loc[index_path]['Cód. Falla'])
        sev = Excel.loc[index_path]['Severidad']
        
        path_Flir_aux = thermal_path  + '/'.join(path_Flir.split('/')[-2:])
        
        if not os.path.isfile(path_Flir_aux):
            print ('No existe la imagen', path_Flir_aux)
            continue
        
        flir = flirimageextractor.FlirImageExtractor()
        
        try:
            flir.process_image(path_Flir_aux)
            I = flirimageextractor.FlirImageExtractor.get_thermal_np(flir)
            w, h = I.shape
            
        except:
            print('No se puede leer la imagen Flir', path_Flir_aux)
            continue
        
        dic_data = flir.get_metadata(path_Flir_aux)
        meas = [s for s in dic_data.keys() if "Meas" in s]
        q_bbox = len(meas)//3 # cada bbox tiene 3 parametros
        
        param_bbox = []
        for num_bbox in range(1, q_bbox + 1):
            # Se guarda los parametros de los boundibox (xmin, ymin, width, height) width = xmax- xmin
            param_bbox.append(list(map(int, dic_data['Meas' + str(num_bbox) + 'Params'].split(' '))))
        
        ##### Save Image and create XML annotations type of fault
        path_save_img = output_path + 'images/' + '_'.join(path_Flir.split('/')[-2:])
        path_save_anns =  output_path + 'anns/' + '_'.join(path_Flir.split('/')[-2:])
        path_save_anns = path_save_anns[:-4] + '.xml'
        
        if not os.path.isfile(path_save_img):
            plt.imsave(path_save_img , I, cmap = 'gray')
        
        #si el archivo ya existe se agregan mas anotaciones
        if os.path.isfile(path_save_anns):
            
            et = ET.parse(path_save_anns)
            root = et.getroot()
            for box in param_bbox:
    
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(cod_falla)
                ET.SubElement(obj, "pose").text = 'Unspecified'
                ET.SubElement(obj, "truncated").text = str(0)
                ET.SubElement(obj, "difficult").text = str(0)
                bx = ET.SubElement(obj, "bndbox")
                ET.SubElement(bx, "xmin").text = str(box[0])
                ET.SubElement(bx, "ymin").text = str(box[1])
                ET.SubElement(bx, "xmax").text = str(box[0] + box[2])
                ET.SubElement(bx, "ymax").text = str(box[1] + box[3])
        
            tree = ET.ElementTree(root)
            tree.write(path_save_anns)      
        
        ## Si no existe se crea desde cero
        else:
            
            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = output_path[:-1]
            ET.SubElement(root, "filename").text = '_'.join(path_Flir.split('/')[-2:])
            ET.SubElement(root, "path").text = path_save_img
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = 'Unknown'
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = str(1)
            ET.SubElement(root, "segmented").text = '0'
        
            for box in param_bbox:
    
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(cod_falla)
                ET.SubElement(obj, "pose").text = 'Unspecified'
                ET.SubElement(obj, "truncated").text = str(0)
                ET.SubElement(obj, "difficult").text = str(0)
                bx = ET.SubElement(obj, "bndbox")
                ET.SubElement(bx, "xmin").text = str(box[0])
                ET.SubElement(bx, "ymin").text = str(box[1])
                ET.SubElement(bx, "xmax").text = str(box[0] + box[2])
                ET.SubElement(bx, "ymax").text = str(box[1] + box[3])
        
            tree = ET.ElementTree(root)
            tree.write(path_save_anns)         
        
        
        
        
        
        
        
        
    
    
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