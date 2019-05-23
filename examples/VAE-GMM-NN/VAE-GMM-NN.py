#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import vae
import gmm
import nn
import serket as srk
import numpy as np
import tensorflow as tf

class NNmodel(nn.NN):
    def model( self, x, y, input_dim, output_dim ):
        hidden_dim = 64
        weight_stddev = 0.1
        
        # layer1
        w1 = tf.Variable(tf.truncated_normal([input_dim, hidden_dim], stddev=weight_stddev), name="w1")
        b1 = tf.Variable(tf.constant(0., shape=[hidden_dim]), name="b1")
        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        
        # layer2
        w2 = tf.Variable(tf.truncated_normal([hidden_dim, output_dim], stddev=weight_stddev), name="w2")
        b2 = tf.Variable(tf.constant(0., shape=[output_dim]), name="b2")
        y2 = tf.matmul(y1, w2) + b2
        
        #loss, 最適化手法の定義
        loss = tf.reduce_sum(tf.square(y2 - y))
        train_step = tf.train.AdamOptimizer()
        
        return loss, train_step

def main():
    obs1 = srk.Observation( np.loadtxt("data1.txt") )
    obs2 = srk.Observation( np.loadtxt("data2.txt") )
    category = np.loadtxt("category.txt")
    
    vae1 = vae.VAE(18, itr=200, batch_size=500)
    gmm1 = gmm.GMM(10, category=category)
    nn1 = NNmodel(itr1=150, itr2=1500, batch_size1=500, batch_size2=500)
    
    vae1.connect( obs1 )
    gmm1.connect( vae1 )
    nn1.connect( gmm1, obs2 )
    
    for i in range(10):
        print( i )
        vae1.update()
        gmm1.update()
        nn1.update()

if __name__=="__main__":
    main()
    
    