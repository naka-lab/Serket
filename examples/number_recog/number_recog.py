#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import word_recog
import mlda
import vae2
import gmm
import TtoT
import numpy as np
import tensorflow as tf
import glob
import logging

hidden_encoder_dim1 = 128
hidden_decoder_dim1 = 128

class vae_model(vae2.VAE):
    def build_encoder(self, x, input_dim, latent_dim):
        h_encoder1 = tf.keras.layers.Dense(hidden_encoder_dim1, activation=tf.nn.relu)(tf.reshape(x,(-1,input_dim[0])))

        mu_encoder = tf.keras.layers.Dense(latent_dim)(h_encoder1)
        logvar_encoder = tf.keras.layers.Dense(latent_dim)(h_encoder1)
        
        return mu_encoder, logvar_encoder
    
    def build_decoder(self, z, input_dim, latent_dim):
        h_decoder1 = tf.keras.layers.Dense(hidden_decoder_dim1, activation=tf.nn.relu)(z)
        x_hat = tf.keras.layers.Dense(input_dim[0])(h_decoder1)

        optimizer = tf.train.AdamOptimizer(0.001)
        
        return x_hat, optimizer

def main():
    logging.basicConfig(level=logging.INFO)

    wavs0 = glob.glob( "speech/0/*.wav" ) # 音声0
    wavs1 = glob.glob( "speech/1/*.wav" ) # 音声1
    wavs2 = glob.glob( "speech/2/*.wav" ) # 音声2
    wavs3 = glob.glob( "speech/3/*.wav" ) # 音声3
    wavs4 = glob.glob( "speech/4/*.wav" ) # 音声4
    wavs5 = glob.glob( "speech/5/*.wav" ) # 音声5
    wavs6 = glob.glob( "speech/6/*.wav" ) # 音声6
    wavs7 = glob.glob( "speech/7/*.wav" ) # 音声7
    wavs8 = glob.glob( "speech/8/*.wav" ) # 音声8
    wavs9 = glob.glob( "speech/9/*.wav" ) # 音声9
    
    wavs = []
    for n in range(300):
        wavs += [ [wavs0[n]], [wavs1[n]], [wavs2[n]], [wavs3[n]], [wavs4[n]], [wavs5[n]], [wavs6[n]], [wavs7[n]], [wavs8[n]], [wavs9[n]] ]

    obs1 = srk.Observation( np.loadtxt("image.txt") )

    category = np.loadtxt( "category.txt" )
    
    speech = word_recog.WordRecog( wavs )
    lda1 = mlda.MLDA( 10, [100], category=category )
    vae1 = vae_model( 10, itr=200, batch_size=500 )
    gmm1 = gmm.GMM( 10, itr=50, category=category )    
    tt = TtoT.TtoT()
    
    lda1.connect( speech )
    vae1.connect( obs1 )
    gmm1.connect( vae1 )
    tt.connect( lda1, gmm1 )

    for it in range(40):
        print(it)
        speech.update()
        lda1.update()
        vae1.update()
        gmm1.update()
        tt.update()

if __name__=="__main__":
    main()


