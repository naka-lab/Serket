---
layout: default
---
## VAE (Variational AutoEncoder)

```
vae.VAE( latent_dim, weight_stddev=0.1, itr=5000, name="vae", hidden_encoder_dim=100,
      　　  hidden_decoder_dim=100, batch_size=None, KL_param=1, mode="learn"  )
```

`vae.VAE` is a module for dimensional compression, and sends compressed latent variables to the connected module.
The variational lower bound of normal VAE is as follows:

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\boldsymbol{o})=-D_{KL}(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{o})||\mathcal{N}(0,\boldsymbol{I}))+\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{o})}[\log{p_{\boldsymbol{\theta}}(\boldsymbol{o}|\boldsymbol{z})}].
$$

On the other hand, in Serket, it is optimized using messages (\\( \boldsymbol{\mu} \\)) received from the connected module as follows:

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\boldsymbol{o})=- \alpha D_{KL}(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{o})||\mathcal{N}(\boldsymbol{\mu},\boldsymbol{I}))+\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{o})}[\log{p_{\boldsymbol{\theta}}(\boldsymbol{o}|\boldsymbol{z})}],
$$

where \\( D_{KL} \\) represents KL divergence and \\( \alpha \\) is the weight of KL divergence.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| latent_dim | int | Number of dimensions of latent variables |
| weight_stddev | float | Standard deviation of weight |
| itr       | int | Number of iteration |
| name      | string | Name of module |
| hidden_encoder_dim | int | Number of nodes in encoder |
| hidden_decoder_dim | int | Number of nodes in decoder |
| batch_size | int | Number of batch size |
| KL_param  | float | Weight of KL divergence |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |


### Methods

- .connect()  
This method connects this module to an observation or a module and constructs the model.
- .update()  
This method estimates model parameters and computes the latent variables.
The module estimates model parameters in "learn" mode and computes the latent variables of novel data using the parameters in "recog" mode.
If training is succeeded, the `module {n} _vae` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.ckpt`: The model parameters.
    - `loss.txt`: The losses computed in the training phase.
    - `x_hat_{mode}.txt`: Reconstructed data by the decoder of VAE.
    - `z_{mode}.txt`: The latent variables computed by the encoder of VAE.  


### Example

```
# import necessary modules
import serket as srk
import vae
import gmm
import numpy as np

data = np.loadtxt( "data.txt" )  # load data
data_category = np.loadtxt( "category.txt" )  # load correct labels

# define the modules
obs = srk.Observation( data )  # send the observation to the connected module
vae1 = vae.VAE( 10 )  # compress to ten dimension
gmm1 = gmm.GMM( 10, catogory=data_category )  # classify into ten classes

# construct the model
vae1.connect( obs )  # connect obs to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1

# optimize the model
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
```
