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
            cv_image = np.array(image)
            if len(cv_image.shape) > 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

            # Apply Otsu's thresholding
            _, thresholded_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Convert the thresholded image back to PIL format
            pil_image = Image.fromarray(thresholded_image)

            processed_images.append(pil_image)

        return processed_images