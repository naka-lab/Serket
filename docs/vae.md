---
layout: default
---
## VAE (Variational AutoEncoder)

```
vae.VAE( latent_dim, weight_stddev=0.1, itr=5000, name="vae", hidden_encoder_dim=100,
      　　  hidden_decoder_dim=100, batch_size=None, KL_param=1, mode="learn"  )
```

VAE is a module for dimensional compression, and sends compressed latent variables to the connected module.
Although the variational lower bound of normal VAE is as follows,

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta,&space;\phi;&space;o)&space;=&space;-D_{KL}(q_{\phi}(z_1|o)||\mathcal{N}(0,&space;I))&plus;\mathbb{E}_{q_{\phi}(z_1|o)}[\log&space;p_{\theta}(o|z_1)]" />

In Serket, it is optimized using messages (<img src="https://latex.codecogs.com/gif.latex?\mu" />) received from the connected module by defining as follows.

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta,&space;\phi;&space;o)&space;=&space;-D_{KL}(q_{\phi}(z_1|o)||\mathcal{N}(\mu,&space;I))&plus;\mathbb{E}_{q_{\phi}(z_1|o)}[\log&space;p_{\theta}(o|z_1)]" />

  
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
| KL_param  | float | Weight for KL divergence |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |

  
### Methods

- .connect()  
This method connects the module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and calculates latent variables and others.
The module estimates model parameters in "learn" mode and predict unknown data in "recog" mode.
If learning is successful, the `module {n} _vae` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.ckpt`: The model parameters are saved.
    - `loss.txt`: The losses calculated at learning are saved
    - `x_hat_{mode}.txt`: Restoration data output from decoder are saved.
    - `z_{mode}.txt`: Latent variables compressed by encoder are saved.