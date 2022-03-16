#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import serket_ros as srkros
import mlda
import CNN
import bow
import rospy
from serket.utils import Buffer

def main():
    rospy.init_node( "image_categorization" )

    obs_img = srkros.ObservationImg( "image" )
    obs_speech = srkros.ObservationString( "google_speech/recres" )

    cnn = CNN.CNNFeatureExtractor( fileames=None )
    cnn_buf = Buffer()

    bow_ = bow.BoW()
    bow_buff = Buffer()

    mlda1 = mlda.MLDA( 3, [1000, 100] )

    cnn.connect( obs_img )
    cnn_buf.connect( cnn )

    bow_.connect( obs_speech )
    bow_buff.connect( bow_ )

    mlda1.connect( cnn_buf, bow_buff )

    n = 0
    for i in range(6):
        print("***", n , "***")
        obs_speech.update()
        obs_img.update()

        bow_.update()
        bow_buff.update()

        cnn.update()
        cnn_buf.update()

        mlda1.update()
        n += 1

if __name__=="__main__":
    main()
    
    