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
import tensorflow as tf

encoder_dim = 128
decoder_dim = 128

class vae_model(vae.VAE):
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
    