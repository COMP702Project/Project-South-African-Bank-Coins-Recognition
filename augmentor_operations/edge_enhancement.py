import Augmentor
import cv2
import numpy as np 

from Augmentor.Operations import Operation
from PIL import Image,ImageEnhance,ImageFilter

class Apply(Operation):
    
    def __init__(self, probability, method):
        Operation.__init__(self, probability)
        self.method = method

    def perform_operation(self, images):
        enhanced_images = []
        for image in images:
            if self.method == "canny":
                enhanced_image = self.apply_canny_edge_detection(image)
            elif self.method == "sobel":
                enhanced_image = self.apply_sobel_edge_detection(image)
            else:
                raise ValueError("Invalid edge detection method")

            enhanced_images.append(enhanced_image)

        return enhanced_images

    def apply_canny_edge_detection(self, image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(cv_image, 100, 200)
        enhanced_cv_image = cv_image.copy()
        enhanced_cv_image[edges != 0] = [255, 0, 0]  # Set edge pixels to red
        enhanced_pil_image = Image.fromarray(cv2.cvtColor(enhanced_cv_image, cv2.COLOR_BGR2RGB))
        return enhanced_pil_image

    def apply_sobel_edge_detection(self, image):
        cv_image = np.array(image)

        # Check if the image has a single channel
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(cv_image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(cv_image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        enhanced_cv_image = cv_image.copy()
        enhanced_cv_image[gradient_magnitude > 50] = [255, 255, 255]  # Enhance edges by setting edge pixels to white
        enhanced_pil_image = Image.fromarray(enhanced_cv_image)
        return enhanced_pil_image