import Augmentor
import cv2
import numpy as np 

from Augmentor.Operations import Operation
from PIL import Image,ImageEnhance,ImageFilter

import pywt

class Apply(Operation):
    def __init__(self, probability, threshold=50, denoising_method='wavelet'):
        Operation.__init__(self, probability)
        self.threshold = threshold
        self.denoising_method = denoising_method

    def wavelet_denoising(self, image):
        # Apply DWT for wavelet denoising
        coeffs = pywt.dwt2(image.astype(np.float32), 'haar')
        denoised_coeffs = tuple(pywt.threshold(c, value=self.threshold, mode='soft') for c in coeffs)
        denoised_image = pywt.idwt2(denoised_coeffs, 'haar')
        return denoised_image.astype(np.uint8)

    def perform_operation(self, images):
        processed_images = []
        for image in images:
            # Convert the binary image to a 3-channel grayscale image
            cv_image = np.array(image.convert('L'))
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

            # Apply denoising based on the selected method
            if self.denoising_method == 'wavelet':
                denoised_image = self.wavelet_denoising(cv_image)
            elif self.denoising_method == 'nlmeans':
                denoised_image = cv2.fastNlMeansDenoising(cv_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
            elif self.denoising_method == 'fourier':
                # Convert image to floating-point format
                cv_image_float = cv_image.astype(np.float32) / 255.0
                # Apply DFT for Fourier transform denoising
                dft = cv2.dft(cv_image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
                # Shift the zero frequency component to the center
                dft_shift = np.fft.fftshift(dft)
                # Set a threshold to remove high-frequency noise
                dft_shift[np.abs(dft_shift) < self.threshold] = 0
                # Inverse shift and transform
                dft_inverse_shift = np.fft.ifftshift(dft_shift)
                denoised_image = cv2.idft(dft_inverse_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
                # Convert image back to uint8 format
                denoised_image = (denoised_image * 255.0).astype(np.uint8)
            else:
                raise ValueError("Invalid denoising method")

            # Perform Sobel edge detection
            sobel_x = cv2.Sobel(denoised_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(denoised_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Threshold the gradient magnitude to create a binary edge map
            _, binary_edge_map = cv2.threshold(gradient_magnitude, self.threshold, 255, cv2.THRESH_BINARY)

            # Bridge broken pixels in the binary edge map
            kernel = np.ones((3, 3), np.uint8)
            bridged_edge_map = cv2.morphologyEx(binary_edge_map, cv2.MORPH_CLOSE, kernel)

            # Resize the bridged edge map to match the dimensions of the input image
            bridged_edge_map = cv2.resize(bridged_edge_map, (cv_image.shape[1], cv_image.shape[0]))

            # Overlay the bridged edge map on the original image
            enhanced_cv_image = cv_image_rgb.copy()
            enhanced_cv_image[bridged_edge_map > 0] = [255, 255, 255]

            # Convert the image back to PIL format
            enhanced_pil_image = Image.fromarray(enhanced_cv_image)
            processed_images.append(enhanced_pil_image)

        return processed_images