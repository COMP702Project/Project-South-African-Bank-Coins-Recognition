import cv2
import numpy as np
import pandas as pd
import os
from Augmentor.Pipeline import Pipeline

def Extract(pipeline,class_lbl):
    data = []
    
    image_paths = [str(image.image_path) for image in pipeline.augmentor_images]  # Get the list of image paths from the pipeline
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Extract the filename from the path
        
        # Extract the class label from the path
        dir_path = os.path.dirname(image_path)
        class_label = class_lbl+os.path.basename(dir_path)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = np.ravel(hu_moments)
        
        row = [image_name, class_label] + hu_moments.tolist()
        data.append(row)
    
    columns = ["Image Name", "Class Label"] + [f"Hu Moment {i}" for i in range(1, 8)]
    df = pd.DataFrame(data, columns=columns)
    
    return df