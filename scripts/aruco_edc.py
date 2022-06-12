#! /usr/bin/env python
import cv2 as cv
import numpy as np

from image_preprocessing import ImagePreprocessing
from feature_detection import ImageFeatureProcessing

class ArucoEDC:
    def __init__(self, img):
        self.img_prep = ImagePreprocessing(2, 8)
        self.ifp = ImageFeatureProcessing(img.shape)

    def detect(self,
               img,
               canny_threshold1,
               canny_threshold2,
               kernel_size1,
               kernel_size2,
               sigma,
               t_blocksize,
               t_C,
               ed_ksize,
               erosion_it,
               dilation_it):
        # CLAHE
        clahe_img = self.img_prep.CLAHE(img)
        clahe_img = clahe_img.round().astype(np.uint8)

        # Gray
        gray = cv.cvtColor(clahe_img, cv.COLOR_BGR2GRAY)

        # Gaussian Blur
        blurred = self.img_prep.gaussian_filter(gray, 3)

        edges = cv.Canny(blurred, canny_threshold1, canny_threshold2)

        blur_hsv_img = cv.GaussianBlur(
            edges, (kernel_size1, kernel_size1), sigma
        )

        thr_img = cv.adaptiveThreshold(
            blur_hsv_img,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            t_blocksize,
            t_C,
        )

        erosion_dilation_kernel = np.ones(
            (ed_ksize, ed_ksize), np.uint8
        )
        erosion_img = cv.erode(
            thr_img, erosion_dilation_kernel, iterations=erosion_it
        )
        morphised_image = cv.dilate(
            erosion_img, erosion_dilation_kernel, iterations=dilation_it
        )
        
        # return 3 channel img
        return cv.cvtColor(morphised_image, cv.COLOR_GRAY2BGR)
