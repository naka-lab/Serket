#!/usr/bin/env python2
from __future__ import print_function, unicode_literals
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def main():
    rospy.init_node('image_sender', anonymous=True)

    pub = rospy.Publisher('image', Image, queue_size=10)
    bridge = CvBridge()


    for i in range(6):
        img = cv2.imread( "images/%03d.png" % i )
        msg = bridge.cv2_to_imgmsg(img, encoding=str("bgr8"))


        raw_input( "Hit enter to publish. " )
        pub.publish( msg )
        print( "published." )

if __name__ == '__main__':
    main()
