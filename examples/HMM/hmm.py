#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import gmm
import serket as srk
import numpy as np
import random
import matplotlib.pyplot as plt

def gen_data():
    N = 100
    rand_gauss = np.random.randn( N, 2 )
    
    mu = np.array( [[0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]])

    correct_class = []    
    for i in range(N):
        if random.random()<1.0:
            rand_gauss[i] = rand_gauss[i]*0.2 + mu[i%4]
        else:
            rand_gauss[i] = rand_gauss[i]*0.3 + mu[i%4]
            
        correct_class.append( i%4 )

    np.savetxt( "data.txt" , rand_gauss )    
    np.savetxt( "correct.txt", correct_class )
    
    plt.scatter( rand_gauss[:,0], rand_gauss[:,1] )
    plt.show()
    


def main():
    # gen_data()
    
    obs = srk.Observation( np.loadtxt("data.txt") )
    
    correct_classes = np.loadtxt( "correct.txt" )
    
    g = gmm.GMM( 4 )
    g.connect( obs )
    g.update()
    
    

if __name__=="__main__":
    main()
    
    