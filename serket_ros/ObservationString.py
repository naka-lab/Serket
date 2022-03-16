#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
from std_msgs.msg import String
import numpy as np
import cv2
from .ObservationBase import SimpeObservationBase

class ObservationString(SimpeObservationBase):
    def __init__( self, topic_name, name="ObservationString", timeout=-1 ):
        super(ObservationString,self).__init__( topic_name=topic_name, topic_type=String, name=name, timeout=timeout )

    def proc_msg(self, msg):
        print(msg.data)


        # Send message.
        return msg.data

