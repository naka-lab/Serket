#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import hac
import mlda

def main():
    wavs = ["0_jackson_0.wav", "0_jackson_1.wav", "0_jackson_2.wav","1_jackson_0.wav", "1_jackson_1.wav", "1_jackson_2.wav", "2_jackson_0.wav", "2_jackson_1.wav", "2_jackson_2.wav"]
    obs1 = hac.HACFeatureExtractor(wavs, [1,1,1])
    

    mlda1 = mlda.MLDA(3, [200], category=[0,0,0,1,1,1,2,2,2])
    
    mlda1.connect( obs1 )
    
    for it in range(1):
        print( it )
        mlda1.update()

if __name__=="__main__":
    main()
    
    