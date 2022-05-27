#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
from std_msgs.msg import String
import numpy as np
import cv2
from .ObservationBase import SimpeObservationBase
import codecs

class ObservationString(SimpeObservationBase):
    def __init__( self, topic_name, name="ObservationString", timeout=-1 ):
        super(ObservationString,self).__init__( topic_name=topic_name, topic_type=String, name=name, timeout=timeout )
        self.__counter = 0

    def proc_msg(self, msg):
        print(msg.data)


        save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )

        with codecs.open( os.path.join(save_dir, "%03d.txt"%self.__counter ), "a", "utf-8" ) as f:
            f.write( msg.data )
            f.write("\n")

        self.__counter += 1

        # Send message.
        return msg.data

