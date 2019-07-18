from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import vae_pixyz
import gmm_pixyz
import serket as srk
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli

hidden_dim = 128

class vae_model(vae_pixyz.VAE):
    def network(self, input_dim, latent_dim):    
        # inference model q(z|x)
        class Inference(Normal):
            def __init__(self):
                super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

                self.fc1 = nn.Linear(input_dim[0], hidden_dim)
                self.fc21 = nn.Linear(hidden_dim, latent_dim)
                self.fc22 = nn.Linear(hidden_dim, latent_dim)

            def forward(self, x):
                h = F.relu(self.fc1(x))
                return {"loc": self.fc21(h), "scale": self.fc22(h)}

        # generative model p(x|z)    
        class Generator(Bernoulli):
            def __init__(self):
                super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

                self.fc1 = nn.Linear(latent_dim, hidden_dim)
                self.fc21 = nn.Linear(hidden_dim, input_dim[0])

            def forward(self, z):
                h = F.relu(self.fc1(z))
                return {"probs": torch.sigmoid(self.fc21(h))}

        q = Inference().to("cuda")
        p = Generator().to("cuda")

        return q, p, optim.Adam, {"lr":1e-3}

def main():
    obs1 = srk.Observation( np.loadtxt("data.txt") )
    category = np.loadtxt("category.txt")

    vae1 = vae_model(10, itr=200, batch_size=500)
    gmm1 = gmm_pixyz.GMM(10, category=category)

    vae1.connect( obs1 )
    gmm1.connect( vae1 )

    for i in range(5):
        print( i )
        vae1.update(i)
        gmm1.update(i)

if __name__=="__main__":
    main()