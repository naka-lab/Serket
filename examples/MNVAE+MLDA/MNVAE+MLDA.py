#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import mnvae
import mlda
import serket as srk
import numpy as np
import tensorflow as tf
import time

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
    
def main():
    img = np.load("vision.npy")
    word = np.loadtxt("word_hist.txt")
    category = np.loadtxt("correct_category.txt")
    
    obs1 = srk.Observation(word)
    obs2 = srk.Observation(img)
    
    n1 = 100
    n2 = 150
    
    mnvae1 = image_mnvae(32, N=n1, rate=0.001, epoch=3000, batch_size=100, KLD_weights=[1,1e+4])
    mlda1 = mlda.MLDA(81, [n1,n2], category=category)

    mnvae1.connect(obs2)
    mlda1.connect(mnvae1, obs1)
    
    for i in range(10):
        print(i)
        
        t = time.time()
        mnvae1.update()
        print(time.time()-t)
        
        t = time.time()
        mlda1.update()
        print(time.time()-t)
        
    print("finish")
        
if __name__=="__main__":
    main()