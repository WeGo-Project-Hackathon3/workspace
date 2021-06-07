#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

#실제로 동작할 때는 이 한글 주석을 전부 지워주세요.
class Foo:
    def __init__(self):
        # 메세지를 numpy로 바꿔주는 친구가 CvBridge
        self.bridge = CvBridge()

        self.img_sub = rospy.Subscriber(
            "/토픽_이름",
            Image,
            self.callback
        )

        self.img_pub = rospy.Publisher(
            "/토픽_이름2",
            Image,
            queue_size=5
        )
        self.img = None

    def callback(self, _imgData):
        self.img = self.bridge.imgmsg_to_cv2(_imgData,"bgr8")
        # 흑백 이미지의 경우
        # gray_image = self.bridge.imgmsg_to_cv2(_imgData, "mono8")
        self.todo()

    def todo(self):
        #To-Do
        # 보낼때도 역시 bridge 사용
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "bgr8"))


if __name__ == "__main__":
    rospy.init_node("Foo")
    rospy.spin()
