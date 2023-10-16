#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:09:29 2023

@author: nitaishah
"""

import pandas as pd
import numpy as np
import imutils
import skimage
import os
from imutils import paths
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import sklearn 
from skimage import io
from PIL import Image


datagen = ImageDataGenerator(
        rotation_range=45, # rotation
        width_shift_range=0.1, # horizontal shift
        height_shift_range=0.1, # vertical shift
        zoom_range=0.1, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2]
)

dataset = []

image_directory = 'Directory of Normal/Defect Original Images'
SIZE = 128
dataset = []
my_images = os.listdir(image_directory)
my_images
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[4] == 'png'):
        image = cv2.imread(image_directory + '/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        
x = np.array(dataset)

i = 0
for batch in datagen.flow(x, batch_size=32,  
                          save_to_dir='Directory to where you want to save the Augmented Images to', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 50:
        break
        

