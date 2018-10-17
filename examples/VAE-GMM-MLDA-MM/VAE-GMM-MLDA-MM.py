#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import vae
import gmm
import mlda
import mm
import serket as srk
import numpy as np


def main():
    obs1 = srk.Observation( np.loadtxt("mnistdata.txt") )
    obs2 = srk.Observation( np.loadtxt("hac.txt") )
    
    category = np.loadtxt("label.txt")
    
    vae1 = vae.VAE(15, itr=150, batch_size=500)
    gmm1 = gmm.GMM(10, category=category)
    mlda1 = mlda.MLDA(10, [200,200], category=category)
    mm1 = mm.MarkovModel()
    
    vae1.connect( obs1 )
    gmm1.connect( vae1 )
    mlda1.connect( obs2, gmm1 )
    mm1.connect( mlda1 )
    
    for it in range(5):
        print( it )
        vae1.update()
        gmm1.update()
        mlda1.update()
        mm1.update()
    


if __name__=="__main__":
    main()
    
    