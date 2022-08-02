#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
from sensor_msgs.msg import Image
import numpy as np
import cv2
from .ObservationBase import SimpeObservationBase

class ObservationImg(SimpeObservationBase):
    def __init__( self, topic_name, name="ObservationImg", timeout=-1 ):
        super(ObservationImg,self).__init__( topic_name=topic_name, topic_type=Image, name=name, timeout=timeout )
        self.__counter = 0

    def proc_msg(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # Save received image.
        save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )
        # Note: OpenCV's default color space is BGR.
        cv2.imwrite( os.path.join( save_dir, "%03d.png"%self.__counter ), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )

        self.__counter += 1

        # Send message.
        return img
