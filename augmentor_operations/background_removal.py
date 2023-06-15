import Augmentor
import cv2
import numpy as np 

from Augmentor.Operations import Operation
from PIL import Image,ImageEnhance,ImageFilter

class Apply(Operation):
    
    def __init__(self, probability):
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        processed_images = []
        for image in images:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Perform GrabCut algorithm for background removal
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (1, 1, cv_image.shape[1] - 1, cv_image.shape[0] - 1)  # Rectangle encompassing the entire image
            cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create a mask based on the final segmentation
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Apply the mask to the original image
            segmented_image = cv_image * mask[:, :, np.newaxis]

            # Convert the image back to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            processed_images.append(pil_image)

        return processed_images