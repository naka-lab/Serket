#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.path.append( "../" )

import os
from nav_msgs.msg import Odometry
import numpy as np
from .ObservationBase import SimpeObservationBase

class ObservationPos(SimpeObservationBase):
    def __init__( self, topic_name, name="ObservationPos", timeout=-1 ):
        super(ObservationPos,self).__init__( topic_name=topic_name, topic_type=Odometry, name=name, timeout=timeout )

    def proc_msg(self, msg):

        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Save received position.
        save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )
        with open( os.path.join( save_dir, "%03d.txt"%len(self.foward_msg) ), 'a' ) as f:
            f.write( "%lf %lf\n"%(msg.pose.pose.position.x, msg.pose.pose.position.y) )

        # Send message.
        return pos
