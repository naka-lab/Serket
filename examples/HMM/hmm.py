#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import gmm
import mm
import serket as srk
import numpy as np

def main():    
    obs = srk.Observation( np.loadtxt("data.txt") )
    correct_classes = np.loadtxt( "correct.txt" )
    
    # GMM単体
    g = gmm.GMM( 4, category=correct_classes )
    g.connect( obs )
    g.update()
    
    # GMMとマルコフモデルを結合したモデル
    g = gmm.GMM( 4, category=correct_classes )
    m = mm.MarkovModel()

    g.connect( obs )
    m.connect( g )
    
    for itr in range(5):
        g.update()
        m.update()

if __name__=="__main__":
    main()
    
    