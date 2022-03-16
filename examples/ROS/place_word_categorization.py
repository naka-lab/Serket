#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import serket_ros as srkros
import mlda
import gmm
import bow
import rospy
from serket.utils import Buffer

def main():
    rospy.init_node( "pos_categorization" )

    obs_pos = srkros.ObservationPos( "odom" )
    obs_speech = srkros.ObservationString( "google_speech/recres" )

    pos_buf = Buffer()

    bow_ = bow.BoW()
    bow_buff = Buffer()

    gmm1 = gmm.GMM( 4 )

    mlda1 = mlda.MLDA( 4, [100, 100] )

    pos_buf.connect( obs_pos )
    gmm1.connect( pos_buf )

    bow_.connect( obs_speech )
    bow_buff.connect( bow_ )

    mlda1.connect( gmm1, bow_buff )

    n = 0
    while not rospy.is_shutdown():
        print("***", n , "***")
        obs_speech.update()
        obs_pos.update()

        bow_.update()
        bow_buff.update()

        pos_buf.update()

        gmm1.update()

        mlda1.update()
        n += 1

if __name__=="__main__":
    main()