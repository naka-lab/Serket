#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import serket_ros as srkros
import mlda
import CNN
import rospy

def main():
    rospy.init_node( "iamage_categorization" )

    obs = srkros.ObservationIMG( "image" )
    cnn = CNN.CNNFeatureExtractor( fileames=None )
    mlda1 = mlda.MLDA( 3, [1000] )

    cnn.connect( obs )
    mlda1.connect( cnn )

    n = 0
    while not rospy.is_shutdown():
        print("***", n , "***")
        obs.update()
        cnn.update()
        mlda1.update()
        n += 1

if __name__=="__main__":
    main()
    
    