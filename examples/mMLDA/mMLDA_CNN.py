#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append("../../")

import mlda
import serket as srk
import numpy as np
import CNN


def main():
    obs1 = CNN.CNNFeatureExtractor( ["images/%03d.png"%i for i in range(6)] )
    obs2 = srk.Observation( np.loadtxt("histogram_w.txt") )

    mlda1 = mlda.MLDA(3, [1000])
    mlda2 = mlda.MLDA(3, [50])
    mlda3 = mlda.MLDA(3, [50,50])
    
    mlda1.connect( obs1 )
    mlda2.connect( obs2 )
    mlda3.connect( mlda1, mlda2 )
    
    for it in range(5):
        mlda1.update()
        mlda2.update()
        mlda3.update()
    


if __name__=="__main__":
    main()
    
    