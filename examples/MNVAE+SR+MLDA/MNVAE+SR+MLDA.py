#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import mnvae
import mlda
import speech_recog
import serket as srk
import numpy as np
import tensorflow as tf
import time
import glob

class image_mnvae(mnvae.MNVAE):
    def build_encoder(self, x, latent_dim):
        h_en = tf.keras.layers.Conv2D(16, 3, 2, padding="same", activation="relu")(x)
        h_en = tf.keras.layers.Conv2D(32, 3, 2, padding="same", activation="relu")(h_en)
        h_en = tf.keras.layers.Conv2D(64, 3, 2, padding="same", activation="relu")(h_en)
    
        self.shape = h_en.shape
        h_en = tf.keras.layers.Flatten()(h_en)
        
        h_en = tf.keras.layers.Dense(latent_dim)(h_en)
        
        return h_en
    
    def build_decoder(self, z):
        h_de = tf.keras.layers.Dense(self.shape[1] * self.shape[2] * self.shape[3], activation="relu")(z)
        h_de = tf.keras.layers.Reshape( (self.shape[1], self.shape[2], self.shape[3]) )(h_de)
        
        h_de = tf.keras.layers.Conv2DTranspose(64, 3, 1, padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(32, 3, 1, padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(16, 3, 1, padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(16, 3, 2, padding="same", activation="relu")(h_de)
        
        logits = tf.keras.layers.Conv2DTranspose(3, 3, 1, padding="same")(h_de)

        optimizer = tf.train.AdamOptimizer()
        
        return logits, optimizer

class tactile_mnvae(mnvae.MNVAE):
    def build_encoder(self, x, latent_dim):
        shape = x.shape
        h_en = tf.keras.layers.Reshape( (1, shape[1], shape[2]) )(x)
        h_en = tf.keras.layers.Conv2D(32, (1,3), (1,2), padding="same", activation="relu")(h_en)
        h_en = tf.keras.layers.Conv2D(64, (1,3), (1,2), padding="same", activation="relu")(h_en)
        h_en = tf.keras.layers.Conv2D(128, (1,3), (1,2), padding="same", activation="relu")(h_en)
        h_en = tf.keras.layers.Conv2D(256, (1,3), (1,2), padding="same", activation="relu")(h_en)
        
        self.shape = h_en.shape
        h_en = tf.keras.layers.Flatten()(h_en)
    
        h_en = tf.keras.layers.Dense(latent_dim)(h_en)
        
        return h_en
    
    def build_decoder(self, z):
        h_de = tf.keras.layers.Dense(self.shape[1] * self.shape[2] * self.shape[3], activation="relu")(z)
        h_de = tf.keras.layers.Reshape( (self.shape[1], self.shape[2], self.shape[3]) )(h_de)
        
        h_de = tf.keras.layers.Conv2DTranspose(256, (1,3), (1,1), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(256, (1,3), (1,2), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(128, (1,3), (1,1), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(128, (1,3), (1,2), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(64, (1,3), (1,1), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(64, (1,3), (1,2), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(32, (1,3), (1,1), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(32, (1,3), (1,2), padding="same", activation="relu")(h_de)
        
        logits = tf.keras.layers.Conv2DTranspose(2, (1,3), (1,1), padding="same")(h_de)
        
        optimizer = tf.train.AdamOptimizer()
        
        return logits, optimizer
   
class audio_mnvae(mnvae.MNVAE):
    def build_encoder(self, x, latent_dim):
        h_en = tf.keras.layers.Conv2D(128, (1,3), (1,2), padding="same", activation="relu")(x)
        h_en = tf.keras.layers.Conv2D(256, (1,3), (1,2), padding="same", activation="relu")(h_en)
        h_en = tf.keras.layers.Conv2D(512, (1,3), (1,2), padding="same", activation="relu")(h_en)
        
        self.shape = h_en.shape
        h_en = tf.keras.layers.Flatten()(h_en)
    
        h_en = tf.keras.layers.Dense(latent_dim)(h_en)
        
        return h_en
    
    def build_decoder(self, z):
        h_de = tf.keras.layers.Dense(self.shape[1] * self.shape[2] * self.shape[3], activation="relu")(z)
        h_de = tf.keras.layers.Reshape( (self.shape[1], self.shape[2], self.shape[3]) )(h_de)
        
        h_de = tf.keras.layers.Conv2DTranspose(512, (1,3), (1,2), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(256, (1,3), (1,2), padding="same", activation="relu")(h_de)
        h_de = tf.keras.layers.Conv2DTranspose(128, (1,3), (1,2), padding="same", activation="relu")(h_de)
        
        logits = tf.keras.layers.Conv2DTranspose(129, (1,3), (1,1), padding="same")(h_de)
        
        optimizer = tf.train.AdamOptimizer()
        
        return logits, optimizer
    
def main():
    wavs = [ glob.glob( "speech/%04d/word/wav/*.wav" % i ) for i in range(499) ]
    img = np.load("vision.npy")
    tac = np.load("tactile.npy")
    audio = np.load("audio.npy")
    category = np.loadtxt("correct_category.txt")
    
    obs1 = srk.Observation( img )
    obs2 = srk.Observation( tac )
    obs3 = srk.Observation( audio )
    
    n1 = 150
    n2 = 100
    n3 = 100
    n4 = 100
    
    speech = speech_recog.SpeechRecog(wavs, nbest=5, itr=200, lmp=[2.0, -20.0])
    mnvae1 = image_mnvae(32, N=n2, rate=0.001, epoch=500, batch_size=100, KLD_weights=[1,1e+4])
    mnvae2 = tactile_mnvae(32, N=n3, rate=0.001, epoch=500, batch_size=100, KLD_weights=[1,1e+3], RE="MSE")
    mnvae3 = audio_mnvae(32, N=n4, rate=0.001, epoch=500, batch_size=100, KLD_weights=[1,1e+3], RE="MSE")
    mlda1 = mlda.MLDA(81, [n1,n2,n3,n4], category=category)
    
    mnvae1.connect( obs1 )
    mnvae2.connect( obs2 )
    mnvae3.connect( obs3 )
    mlda1.connect( speech, mnvae1, mnvae2, mnvae3 )
    
    for i in range(10):
        print(i)
        
        t = time.time()
        speech.update()
        print(time.time() - t)
        
        t = time.time()
        mnvae1.update()
        print(time.time()-t)
        
        t = time.time()
        mnvae2.update()
        print(time.time()-t)
        
        t = time.time()
        mnvae3.update()
        print(time.time()-t)
        
        t = time.time()
        mlda1.update()
        print(time.time()-t)
        
    print("finish")
        
if __name__=="__main__":
    main()