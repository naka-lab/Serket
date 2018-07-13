#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import mlda
import serket as srk
import numpy as np


def main():
    obs1 = srk.Observation( np.loadtxt("dsift.txt") )       # 視覚
    obs2 = srk.Observation( np.loadtxt("mfcc.txt") )        # 聴覚
    obs3 = srk.Observation( np.loadtxt("tactile.txt") )     # 触覚
    obs4 = srk.Observation( np.loadtxt("angle.txt") )       # 関節角
    
    object_category = np.loadtxt( "object_category.txt" )
    motion_category = np.loadtxt( "motion_category.txt" )

    mlda1 = mlda.MLDA(10, [200,200,200], category=object_category)
    mlda2 = mlda.MLDA(10, [200], category=motion_category)
    mlda3 = mlda.MLDA(10, [100,100])
    
    mlda1.connect( obs1, obs2, obs3 )
    mlda2.connect( obs4 )
    mlda3.connect( mlda1, mlda2 )
    
    for it in range(5):
        print( it )
        mlda1.update()
        mlda2.update()
        mlda3.update()
    


if __name__=="__main__":
    main()
    
    