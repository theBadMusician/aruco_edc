#! /usr/bin/env python

# import debugpy
# print("Waiting for VSCode debugger...")
# debugpy.listen(5678)
# debugpy.wait_for_client()

import rospy
import rospkg
import sys
import os

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import dynamic_reconfigure.client

from aruco_edc import ArucoEDC

import numpy as np
import cv2 as cv

class ArucoEDCNode():
    """Performs image preprocessing (duh...)
    """

    def __init__(self, image_topic):
        rospy.init_node('aruco_edc_node')
        self.rospack = rospkg.RosPack()
        
        # ns = "/" + (image_topic.split("/"))[1]

        # self.zedSub                 = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size= 1)

        # self.CLAHEPub               = rospy.Publisher('/cv/aruco_edc/CLAHE' + ns, Image, queue_size= 1)
        # self.single_CLAHEPub        = rospy.Publisher('/cv/aruco_edc/CLAHE_single' + ns, Image, queue_size= 1)
        # self.GWPub                  = rospy.Publisher('/cv/aruco_edc/GW' + ns, Image, queue_size= 1)
        
        self.ros_rate = rospy.Rate(10.0)

        self.orig_img_pub        = rospy.Publisher('/cv/aruco_edc/orig', Image, queue_size= 1)
        self.proc_img_pub        = rospy.Publisher('/cv/aruco_edc/proc', Image, queue_size= 1)

        self.bridge = CvBridge()
        
        # First initialization of image shape
        # first_image_msg = rospy.wait_for_message(image_topic, Image)

        aruco_edc_pkg_path = self.rospack.get_path('aruco_edc')
        pic_path = aruco_edc_pkg_path + "/aruco_dataset/f0.jpg"
        self.cv_image = cv.imread(pic_path)

        # self.cv_image = self.bridge.imgmsg_to_cv2(first_image_msg, "passthrough")
        self.image_shape = self.cv_image.shape
        self.aruco_edc = ArucoEDC(self.cv_image)

        # ======================================> PARAMS
        # Canny params
        self.canny_threshold1 = 100
        self.canny_threshold2 = 200
        self.canny_aperture = 3

        # HSV params
        self.hsv_params = [0,
                           179,
                           0,
                           255,
                           0,
                           255]

        # Blur params
        self.ksize1 = 7
        self.ksize2 = 7
        self.sigma = 0.8

        # Thresholding params
        self.thresholding_blocksize = 11
        self.thresholding_C = 2

        # Erosion and dilation params
        self.erosion_dilation_ksize = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 1
        self.noise_rm_params = [self.ksize1, self.ksize2, self.sigma, self.thresholding_blocksize, self.thresholding_C, self.erosion_dilation_ksize, self.erosion_iterations, self.dilation_iterations]

        self.dynam_client = dynamic_reconfigure.client.Client("/aruco_edc_cfg/aruco_edc_cfg", config_callback=self.dynam_reconfigure_callback)


    def cv_image_publisher(self, publisher, image, msg_encoding="bgr8"):
        """
        Takes a cv::Mat image object, converts it into a ROS Image message type, and publishes it using the specified publisher.
        """
        msgified_img = self.bridge.cv2_to_imgmsg(image, encoding=msg_encoding)
        publisher.publish(msgified_img)
    
    def spin(self):
        while not rospy.is_shutdown():
            if self.cv_image is None:
                continue

            self.cv_image_publisher(self.orig_img_pub, self.cv_image, "bgr8")

            try:
                self.proc_img = self.aruco_edc.detect(self.cv_image,
                                                    self.canny_threshold1,
                                                    self.canny_threshold2,
                                                    self.ksize1,
                                                    self.ksize2,
                                                    self.sigma,
                                                    self.thresholding_blocksize,
                                                    self.thresholding_C, 
                                                    self.erosion_dilation_ksize, 
                                                    self.erosion_iterations,
                                                    self.dilation_iterations)

                self.cv_image_publisher(self.proc_img_pub, self.proc_img, "bgr8")
                rospy.loginfo("Executed ArUco Detection...")
        
            except Exception:
                pass    
            
            self.ros_rate.sleep()

    def dynam_reconfigure_callback(self, config):
        self.canny_threshold1 = config.canny_threshold1
        self.canny_threshold2 = config.canny_threshold2
        self.canny_aperture = config.canny_aperture_size

        self.hsv_params[0] = config.hsv_hue_min
        self.hsv_params[1] = config.hsv_hue_max
        self.hsv_params[2] = config.hsv_sat_min
        self.hsv_params[3] = config.hsv_sat_max
        self.hsv_params[4] = config.hsv_val_min
        self.hsv_params[5] = config.hsv_val_max

        self.ksize1 = config.ksize1
        self.ksize2 = config.ksize2
        self.sigma = config.sigma

        self.thresholding_blocksize = config.blocksize
        self.thresholding_C = config.C

        self.erosion_dilation_ksize = config.ed_ksize
        self.erosion_iterations = config.erosion_iterations
        self.dilation_iterations = config.dilation_iterations

        self.noise_rm_params = [self.ksize1, self.ksize2, self.sigma, self.thresholding_blocksize, self.thresholding_C, self.erosion_dilation_ksize, self.erosion_iterations, self.dilation_iterations]


if __name__ == '__main__':
    try:
        image_topic = sys.argv[1]
        aruco_edc_node = ArucoEDCNode(image_topic)
        aruco_edc_node.spin()

    except rospy.ROSInterruptException:
        pass