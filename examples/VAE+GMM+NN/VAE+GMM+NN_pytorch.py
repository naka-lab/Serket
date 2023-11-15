#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import torch_vae
import torch_nn
import gmm
import serket as srk
import numpy as np
import torch
import torch.nn as nn
import mlda

encoder_dim = 128
decoder_dim = 128

class vae_model(torch_vae.TorchVAE):
    encoder_dim = 128
    decoder_dim = 128

    def build_encoder(self, input_dim, latent_dim):
        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.fc = nn.Linear(input_dim[0], encoder_dim)
                self.fc_mu = nn.Linear(encoder_dim, latent_dim)
                self.fc_var = nn.Linear(encoder_dim, latent_dim)
            
            def forward(self, x):
                h = torch.relu(self.fc(x))
                mu = self.fc_mu(h) 
                log_var = self.fc_var(h)
                return mu, log_var
        return Encoder()
    
    def build_decoder(self, input_dim, latent_dim):
        model = nn.Sequential(
            nn.Linear( latent_dim, decoder_dim ),
            nn.ReLU(),
            nn.Linear( decoder_dim, input_dim[0] ),
            nn.Sigmoid()
        )
        return model

class NN_model(torch_nn.TorchNN):
    def build_model( self, input_dim, output_dim ):
        h_dim = 64

        model = nn.Sequential(
            nn.Linear( input_dim, h_dim ),
            nn.ReLU(),
            nn.Linear( h_dim, output_dim ),
            nn.Sigmoid()
        )
        return model


def main():
    obs1 = srk.Observation( np.loadtxt("data/data1.txt") )
    obs2 = srk.Observation( np.loadtxt("data/data2.txt") )
    category = np.loadtxt("data/category.txt")
    
    vae1 = vae_model(784, 10, device="cpu", epoch=200, batch_size=500)
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
    