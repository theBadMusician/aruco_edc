#! /usr/bin/env python
from xml.dom import HierarchyRequestErr
import cv2 as cv
import numpy as np
import traceback

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
        contours, hierarchy = cv.findContours(
            morphised_image, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE
        )
        # contours, hierarchy = cv.findContours(morphised_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        print(contours)

        filtered_contours = []
        try:
            num_of_contours = len(contours)

            for cnt_idx in range(num_of_contours):
                cnt = contours[cnt_idx]
                cnt_area = cv.contourArea(cnt)
                
                # Filters out contours with less-than-predefined threshold area 
                if cnt_area < 1200:
                    filtered_contours.append(False)
                else:
                    # Filters out contours with less-than-predefined threshold perimeter
                    if cv.arcLength(cnt, True) > 20:
                        filtered_contours.append(True)
                    else:
                        filtered_contours.append(False)
        except Exception:
            print(traceback.format_exc())
            pass

        contours_array = np.array(contours)
        contours_filtered = contours_array[filtered_contours]

        approx_cnts = []
        for cnt in contours_filtered:
            # approx_cnt = cv.approxPolyDP(cnt, 0.5, True) 
            # approx_cnts.append(approx_cnt)
            peri = cv.arcLength(cnt, True)
            corners = cv.approxPolyDP(cnt, 0.04 * peri, True)
            approx_cnts.append(corners)
        

        cnt_image = cv.drawContours(clahe_img, approx_cnts, -1, (0,0,255), 2)
        
        # cv.polylines(clahe_img, approx_cnts, True, (0,0,255), 1, cv.LINE_AA)
        # return 3 channel img
        return clahe_img, cv.cvtColor(morphised_image, cv.COLOR_GRAY2BGR)
