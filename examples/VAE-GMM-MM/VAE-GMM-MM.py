#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import vae
import gmm
import mm
import numpy as np


def main():
    obs = srk.Observation( np.loadtxt("data.txt") )
    data_category = np.loadtxt( "category.txt" )

    vae1 = vae.VAE( 18, itr=200, batch_size=500 )
    gmm1 = gmm.GMM( 10, category=data_category )
    mm1 = mm.MarkovModel()

    vae1.connect( obs )
    gmm1.connect( vae1 )
    mm1.connect( gmm1 )

    for i in range(5):
        print( i )
        vae1.update()
        gmm1.update()
        mm1.update()



if __name__=="__main__":
    main()
