#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
import serket as srk
try:
    import Queue as queue
except ModuleNotFoundError:
    import queue
from nav_msgs.msg import Odometry
import numpy as np
import rospy

class ObservationPos(srk.Module):
    def __init__( self, topic_name, name="ObservationPos", timeout=-1 ):
        super(ObservationPos,self).__init__( name=name, learnable=False )
        self.msg_que = queue.Queue()
        self.foward_msg = []
        self.timeout = timeout

        rospy.Subscriber( topic_name, Odometry, self.msg_callback )

    def msg_callback(self, msg ):
        self.msg_que.put( msg )

    def update(self):
        try:
            msg = self.msg_que.get()
        except queue.Empty:
            return False

        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Save received position.
        save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )
        with open( os.path.join( save_dir, "%03d.txt"%len(self.foward_msg) ), 'w' ) as f:
            f.write( str(pos) )

        # Send message.
        self.foward_msg.append( pos )
        self.set_forward_msg( self.foward_msg )
