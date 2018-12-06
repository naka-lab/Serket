---
layout: default
---
## VAE + GMM
We construct a model integrating VAE and GMM, and do mutual learning of dimension compression by VAE and unsupervised classification by GMM.

### Data
We use [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
The number of data is 3000.
MNIST dataset is handwritten digit image data.

### Model
VAEでは，観測 \\( \boldsymbol{o} \\) がエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_1 \\) に圧縮される．
そして，潜在変数 \\( \boldsymbol{z}_1 \\) がデコーダーにあたるニューラルネットを通して元の次元に復元され，その値と観測 \\( \boldsymbol{o} \\) 同じになるように学習される．
VAEは，このようにして圧縮された潜在変数 \\( \boldsymbol{z}_1 \\) をGMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_1 \\) を分類し，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
通常VAEの変分下限は次式で表される．

In VAE, the observations \\( \boldsymbol{o} \\) are compressed into the latent variables \\( \boldsymbol{z}_1 \\) of an arbitrary number of dimensions through the neural net equivalent of the encoder.
Then, the latent variables are restored to the original number of dimensions through the neural net equivalent of the decoder, and learned so that the values and observations are the same.
VAE sends the latent variables \\( \boldsymbol{z}_1 \\) thus compressed to GMM.
GMM classifies the latent variables \\( \boldsymbol{z}_1 \\) received from VAE and sends the means \\( \boldsymbol{\mu} \\) of the distributions of classes in which each data is classified to VAE.
The variational lower bound of normal VAE is as follows.

$$
\mathcal{L}( \boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{o} ) = -D_{KL} ( q_{ \boldsymbol{\phi} }( \boldsymbol{z}_1 \mid \boldsymbol{o} ) \| \mathcal{N} ( 0, \boldsymbol{I} ) ) + \mathbb{E}_{ q_{ \boldsymbol{\phi} }( \boldsymbol{z}_1 \mid  \boldsymbol{o} ) } [ \log{ p_{ \boldsymbol{\theta} } ( \boldsymbol{o} \mid \boldsymbol{z}_1 ) } ]
$$

Serketでは，GMMでの分類の影響を受けるため，データが分類されたクラスの平均 \\( \mu \\) を用いて変分下限を以下のように定義する．

In Serket, in order to be affected by the classification in GMM, we define the variational lower bound as follows using the means \\( \boldsymbol{\mu} \\) of classes in which each data is classified.

$$
\mathcal{L}( \boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{o} ) = - \alpha D_{KL} ( q_{ \boldsymbol{\phi} } ( \boldsymbol{z}_1 \mid \boldsymbol{o} ) \| \mathcal{N} ( \boldsymbol{\mu}, \boldsymbol{I} ) ) + \mathbb{E}_{ q_{ \boldsymbol{\phi} } ( \boldsymbol{z}_1 \mid \boldsymbol{o} ) } [ \log{ p_{ \boldsymbol{\theta} } ( \boldsymbol{o} \mid \boldsymbol{z}_1 ) } ]
$$

ただし， \\( D_{KL} \\) はKLダイバージェンスを表しており，\\( \alpha \\) はKLダイバージェンスの重みである.
これにより，GMMによって同じクラスに分類されたデータの潜在変数 \\( \boldsymbol{z}_1 \\) は似た値を持つこととなり，分類に適した潜在空間が学習される．

Where \\( D_{KL} \\) represents KL divergence and \\( \alpha \\) is the weight of KL divergence .
This time we use \\( \alpha = 1 \\).
As a result, the latent variables \\( \boldsymbol{z}_1 \\) of data classified into the same class by GMM have similar values, and the latent space suitable for the classification is learned.

<div align="center">
<img src="img/vae-gmm/vae-gmm.png" width="650px">
</div>

### Codes
Firstly, we import the necessary modules.

```
import serket as srk
import vae
import gmm
import numpy as np
```

Secondly, we load data and correct labels.
The data are sent as observations to the connected module by `srk.Observation`.

```
obs = srk.Observation( np.loadtxt( "data.txt" ) )
data_category = np.loadtxt( "category.txt" )
```
Thirdly, we define each module.
We define VAE that compresses the data into 18 dimensions, whose epoch number is 200 and batch size is 500.
We define GMM that classifies the data into ten classes and give `data_category` as correct labels.

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
```

Fourthly, we connect the modules and construct the model.

```
vae1.connect( obs )  # connect obs to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
```

Finallly, we optimize the whole model by repeatedly updating the parameters of each module and exchanging messages.

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
```

### Result
If training the model is successful, `module001_vae` and `module002_gmm` directories are created.
The parameters of each module, probabilities, accuracy, and so on are stored in each directory.
The compressed latent variables are stored in `z_learn.txt` in `module001_vae`.
An example of a graph plotting the latent variables \\( \boldsymbol{z}_1 \\) compressed into two dimensions by principal component analysis is shown below.

<div align="center">
<img src="img/vae-gmm/pca.png" width="600px">
</div>

Data points that are the same class are widely dispersed in the space before optimization, whereas they have similar values for each class after optimization.
It is confirmed that the latent space suitable for the classification is learned by exchanging messages.
The result and the accuracy of the classification are stored in `module002_gmm`.
The indexes of classes in which each data is classified are saved in `class_learn.txt`, and the classification accuracy is saved in `acc_learn.txt`.
