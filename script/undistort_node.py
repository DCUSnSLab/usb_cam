#!/usr/bin/env python3
import rospy
import cv2
import yaml
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageUndistorter:
    def __init__(self):
        # 파라미터 읽기
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.undistorted_topic = rospy.get_param('~undistorted_topic', '/camera/image_undistorted')
        self.camera_info_path = rospy.get_param('~camera_info_path', '')

        # 카메라 파라미터 로드
        with open(self.camera_info_path, 'r') as f:
            info = yaml.safe_load(f)
        K = np.array(info['camera_matrix']['data']).reshape((3, 3))
        D = np.array(info['distortion_coefficients']['data'])
        self.width = info['image_width']
        self.height = info['image_height']

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, None, K, (self.width, self.height), cv2.CV_16SC2
        )

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.callback, queue_size=1)
        rospy.loginfo("Image undistorter ready. Subscribing to: %s, publishing: %s", self.image_topic, self.undistorted_topic)

    def callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            undistorted = cv2.remap(cv_img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            out_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
        except Exception as e:
            rospy.logerr("Error in undistortion: %s", e)

if __name__ == '__main__':
    rospy.init_node('image_undistorter')
    ImageUndistorter()
    rospy.spin()
