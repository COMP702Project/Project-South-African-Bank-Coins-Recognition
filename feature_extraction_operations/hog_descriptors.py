import cv2
import numpy as np
import pandas as pd
import os
from Augmentor.Pipeline import Pipeline

def hog_descriptor(image):

    # values for paramters
    # mostly default values as the documentation recommends
    win_size = (225, 225) 
    block_size=(10, 10) 
    block_stride=(5, 5) 
    cell_size=(10, 10)
    nbins=9
    derivAperture = 1
    win_sigma=-1
    histogramNormType = 1
    threshold_L2hys=0.2
    gamma_correction=False
    nlevels=64
    useSignedGradients=True

    hog = cv2.HOGDescriptor(win_size, 
                        block_size, 
                        block_stride, 
                        cell_size,
                        nbins,
                        derivAperture,
                        win_sigma,
                        histogramNormType,
                        threshold_L2hys,
                        gamma_correction,
                        nlevels,
                        useSignedGradients)


    descriptor = hog.compute(image)
    return descriptor


def Extract(pipeline,class_lbl):
    
    descriptors = []
    
    image_paths = [str(image.image_path) for image in pipeline.augmentor_images]  # Get the list of image paths from the pipeline
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Extract the filename from the path
        
        # Extract the class label from the path
        dir_path = os.path.dirname(image_path)
        class_label = class_lbl+os.path.basename(dir_path)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        descriptor = hog_descriptor(image)
        
        row = [image_name, class_label] + descriptor
        descriptors.append(row)
    
    columns = ["Image Name", "Class Label"] + [f"Hog descriptor {i+1}" for i in range(len(descriptor))]
    df = pd.DataFrame(descriptors, columns=columns)
    
    return df
        
