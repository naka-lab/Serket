#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import vae
import serket as srk
import numpy as np


class VAE(srk.Module):
    def __init__(self, latent_dim, weight_stddev=0.1, itr=5000, name="vae", hidden_encoder_dim=100, hidden_decoder_dim=100, batch_size=None ):
        super(VAE, self).__init__(name, True)
        self.__itr = itr
        self.__latent_dim = latent_dim
        self.__weight_stddev = weight_stddev
        self.__hidden_encoder_dim = hidden_encoder_dim
        self.__hidden_decoder_dim = hidden_decoder_dim
        self.__batch_size = batch_size

    def update(self):
        data = self.get_observations()
        mu_prior = self.get_backward_msg() # P(z|d)

        N = len( data[0] )  # データ数

        # backward messageがまだ計算されていないときは一様分布にする
        if mu_prior is None:
            mu_prior = np.zeros( (N, self.__latent_dim) )

        data[0] = np.array( data[0], dtype=np.float32 )


        # VAE学習
        z, x = vae.train( data[0], self.__latent_dim, self.__weight_stddev, self.__itr, self.get_name(), mu_prior=mu_prior, hidden_encoder_dim=self.__hidden_encoder_dim, hidden_decoder_dim=self.__hidden_encoder_dim, batch_size=self.__batch_size )

        # メッセージの送信
        self.set_forward_msg( z )
        self.send_backward_msgs( x )
