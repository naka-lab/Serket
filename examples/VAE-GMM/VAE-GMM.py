#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import vae
import gmm
import serket as srk
import numpy as np


def main():
    obs1 = srk.Observation( np.loadtxt("mnist.txt") )
    category = np.loadtxt("mnistlabel.txt")
    
    vae1 = vae.VAE(10, itr=4000)
    gmm1 = gmm.GMM(10, category=category)
    
    vae1.connect( obs1 )
    gmm1.connect( vae1 )
    
    for it in range(3):
        print( it )
        vae1.update()
        gmm1.update()
    


if __name__=="__main__":
    main()
    
    