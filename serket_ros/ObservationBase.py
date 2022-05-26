#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
from abc import ABCMeta, abstractmethod
import serket as srk
try:
    import Queue as queue
except ModuleNotFoundError:
    import queue
import numpy as np
import cv2
import rospy

class SimpeObservationBase(srk.Module):
    __metaclass__ = ABCMeta
    def __init__( self, topic_name, topic_type, name, timeout, qsize=1 ):
        super(SimpeObservationBase,self).__init__( name=name, learnable=False )
        self.msg_que = queue.Queue()
        self.foward_msg = []
        self.timeout = timeout
        self.qsize = qsize
        rospy.Subscriber( topic_name, topic_type, self.msg_callback )

    def msg_callback(self, msg ):
        # qsizeだけ保持する
        while self.msg_que.qsize() > self.qsize-1:
            self.msg_que.get_nowait()

        self.msg_que.put( msg )

    def update(self):
        try:
            msg = self.msg_que.get(self.timeout)
        except queue.Empty:
            return False

        msg = self.proc_msg( msg )

        # Send message.
        self.set_forward_msg( [msg] )

    @abstractmethod
    def proc_msg( self, msg ):
        pass
