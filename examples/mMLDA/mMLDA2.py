#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append("../../")

import mlda
import serket as srk
import numpy as np


def main():
    obs1 = srk.Observation( np.loadtxt("data000.txt") )
    obs2 = srk.Observation( np.loadtxt("data001.txt") ) 
    obs3 = srk.Observation( np.loadtxt("data002.txt") )
    obs4 = srk.Observation( np.loadtxt("data003.txt") )
    obs5 = srk.Observation( np.loadtxt("data004.txt") )
    
    
    category = np.loadtxt( "data_category.txt" )

    mlda1 = mlda.MLDA(10, [100], category=category)
    mlda2 = mlda.MLDA(10, [100,100], category=category)
    mlda3 = mlda.MLDA(10, [100,100], category=category)
    mlda4 = mlda.MLDA(10, [100,100], category=category)
    mlda5 = mlda.MLDA(10, [100,100], category=category)
    
    mlda1.connect( obs1 )
    mlda2.connect( mlda1, obs2 )
    mlda3.connect( mlda2, obs3 )
    mlda4.connect( mlda3, obs4 )
    mlda5.connect( mlda4, obs5 )
    
    for it in range(5):
        mlda1.update()
        mlda2.update()
        mlda3.update()
        mlda4.update()
        mlda5.update()
    


if __name__=="__main__":
    main()
    
    