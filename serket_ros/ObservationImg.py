#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
import serket as srk
try:
    import Queue as queue
except:
    import queue
from sensor_msgs.msg import Image
import numpy as np
import cv2
import rospy

class ObservationImg(srk.Module):
    def __init__( self, topic_name, name="ObservationImg", timeout=-1 ):
        super(ObservationImg,self).__init__( name=name, learnable=False )
        self.msg_que = queue.Queue()
        self.foward_msg = []
        self.timeout = timeout

        rospy.Subscriber( topic_name, Image, self.msg_callback )

    def msg_callback(self, msg ):
        self.msg_que.put( msg )

    def update(self):
        try:
            msg = self.msg_que.get()
        except queue.Empty:
            return False

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # Save received image.
        save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )
        cv2.imwrite( os.path.join( save_dir, "%03d.png"%len(self.foward_msg) ), img )

        # Send message.
        self.foward_msg.append( img )
        self.set_forward_msg( self.foward_msg )
