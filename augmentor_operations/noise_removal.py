import Augmentor
import cv2
import numpy as np 

from Augmentor.Operations import Operation
from PIL import Image,ImageEnhance,ImageFilter


class Apply(Operation):
    def __init__(self, noise_probability, filter_type):
        Operation.__init__(self, noise_probability)
        self.filter_type = filter_type

    def perform_operation(self, images):
        denoised_images = []
        for image in images:
            if self.filter_type == "median":
                denoised_image = image.filter(ImageFilter.MedianFilter(size=3))
            elif self.filter_type == "gaussian":
                denoised_image = image.filter(ImageFilter.GaussianBlur(radius=2))
            elif self.filter_type == "mean":
                denoised_image = image.filter(ImageFilter.BoxBlur(radius=2))
            elif self.filter_type == "laplacian":
                _ = cv2.Laplacian(np.array(image), cv2.CV_8U)
                denoised_image = Image.fromarray(_)
            else:
                raise ValueError("Invalid filter type")

            denoised_images.append(denoised_image)
        return denoised_images