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

encoder_dim = 128
decoder_dim = 128

class vae_model(vae.VAE):
    encoder_dim = 128
    decoder_dim = 128
    def build_encoder(self, x, latent_dim):
        h_encoder = tf.keras.layers.Dense(encoder_dim, activation="relu")(x)

        mu = tf.keras.layers.Dense(latent_dim)(h_encoder)
        logvar = tf.keras.layers.Dense(latent_dim)(h_encoder)
        
        return mu, logvar
    
    def build_decoder(self, z):
        h_decoder = tf.keras.layers.Dense(decoder_dim, activation="relu")(z)
        logits = tf.keras.layers.Dense(784)(h_decoder)

        optimizer = tf.train.AdamOptimizer()
        
        return logits, optimizer

class NN_model(nn.NN):
    def model( self, x, y, input_dim, output_dim ):
        h_dim = 64
        
        # hidden layer
        h = tf.keras.layers.Dense(h_dim, activation="relu")(x)
        
        # output layer
        yy = tf.keras.layers.Dense(output_dim)(h)
        
        #loss, 最適化手法の定義
        loss = tf.reduce_mean( tf.reduce_sum(tf.square(yy - y), axis=1) )
        train_step = tf.train.AdamOptimizer()
        
        return loss, train_step

def main():
    obs1 = srk.Observation( np.loadtxt("data1.txt") )
    obs2 = srk.Observation( np.loadtxt("data2.txt") )
    category = np.loadtxt("category.txt")
    
    vae1 = vae_model(10, itr=200, batch_size=500)
    gmm1 = gmm.GMM(10, category=category)
    nn1 = NN_model(itr1=500, itr2=2000, batch_size1=500, batch_size2=500)
    
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
    