import cv2
import numpy as np
import pandas as pd
import os
from Augmentor.Pipeline import Pipeline
from sklearn.decomposition import PCA

def Extract(pipeline, class_lbl, num_components):
    data = []
    
    image_paths = [str(image.image_path) for image in pipeline.augmentor_images]  # Get the list of image paths from the pipeline
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    
    # Initialize the maximum number of SIFT scores
    max_sift_scores = 0
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Extract the filename from the path
        
        # Extract the class label from the path
        dir_path = os.path.dirname(image_path)
        class_label = class_lbl + os.path.basename(dir_path)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect keypoints and compute descriptors using SIFT
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=num_components)
        descriptors = pca.fit_transform(descriptors)
        
        # Flatten the descriptors to a 1D array
        sift_scores = descriptors.flatten()
        
        max_sift_scores = max(max_sift_scores, len(sift_scores))
        
        row = [image_name, class_label] + sift_scores.tolist()
        data.append(row)
    
    columns = ["Image Name", "Class Label"] + [f"SIFT Score {i}" for i in range(1, max_sift_scores + 1)]
    df = pd.DataFrame(data, columns=columns)
    
    return df