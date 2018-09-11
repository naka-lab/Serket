#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import vae
import gmm
import mm
import serket as srk
import numpy as np


def main():
    obs1 = srk.Observation( np.loadtxt("mnistdata.txt") )
    category = np.loadtxt("mnistlabel.txt")
    
    vae1 = vae.VAE(15, itr=200, batch_size=500)
    gmm1 = gmm.GMM(10, category=category)
    mm1 = mm.MarkovModel()
    
    vae1.connect( obs1 )
    gmm1.connect( vae1 )
    mm1.connect( gmm1 )
    
    for it in range(5):
        print( it )
        vae1.update()
        gmm1.update()
        mm1.update()
    


if __name__=="__main__":
    main()
    
    