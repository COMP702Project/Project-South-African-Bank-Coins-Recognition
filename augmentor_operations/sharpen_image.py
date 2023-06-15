import Augmentor
import cv2
import numpy as np 

from Augmentor.Operations import Operation
from PIL import Image,ImageEnhance,ImageFilter


class Apply(Operation):
    
    def __init__(self, probability, level):
        Operation.__init__(self, probability)
        self.level = level

    def perform_operation(self, images):
        sharpened_images = []
        for image in images:
            enhancer = ImageEnhance.Sharpness(image)
            sharpened_image = enhancer.enhance(self.level)
            sharpened_images.append(sharpened_image)
        return sharpened_images