---
layout: default
---
## VAE + GMM + MM
HMM (hidden Markov model) can be constructed by combining GMM (Gaussian mixture model) and MM (Markov model).
Here, we extend the GMM+VAE, which enables unsupervised classification using dimensional compression, and construct a model of unsupervised classification that can learn transition rules by integrating VAE, GMM, and MM. 

### Data
We use handwritten digit image data [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
The number of data is 3000.
In order to learn the transition rule, we sort them in ascending order like 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, \\( \cdots \\).

### Model

<!--
VAEは，観測 \\( \boldsymbol{o} \\) をエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_ 1 \\) に圧縮し，GMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_ 1 \\) を分類し，\\( t \\) 番目のデータがクラス \\( z_ {2,t} \\) に分類される確率 \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) をMMへ送信，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
VAEは，\\( \boldsymbol{\mu} \\) を用いることでGMMの分類に適した潜在空間が学習する．
MMは，送られてきた確率 \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) を用いて繰り返しサンプリングを行い，次のように遷移回数をカウントする．
-->

VAE compresses the observations \\( \boldsymbol{o} \\) into arbitrary dimensional latent variables \\( \boldsymbol{z}_ 1 \\)  through the neural network called encoder and sends them to GMM.
GMM classifies the latent variables \\( \boldsymbol{z}_ 1 \\) recieved from VAE, and then sends the probabilities \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) that the t-th data is classified into the class \\( z_ {2,t} \\) to MM. 
At the same time, GMM sends the means \\( \boldsymbol{\mu} \\) of the distributions of the classes, into which each data is classified, to VAE.
VAE learns the latent space suitable for the classification of GMM by using \\( \mu \\).
Moreover, transition rules are learned in the MM. 
The latent variables \\(z_2)\\ are  repeatedly sampled using the received probabilities \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) and the number of transitions is counted as follows: 

$$
\begin{align}
&z'_ 2 \sim P(z_{2,t} \mid \boldsymbol{z}_{1,t})\\
&z_2 \sim P(z_{2,t+1} \mid \boldsymbol{z}_{1,t+1})\\
&N_{z'_ 2,z_2}++. 
\end{align}
$$

<!--
この値から遷移確率 \\( P(z_ 2 \mid z'_ 2) \\) は次のように計算することができる．
-->

The transition probabilities \\( P(z_ 2 \mid z'_ 2) \\) can be computed from these values as follows: 

$$
P(z_2 \mid z'_ 2) = \frac{N_{z'_ 2,z_2} + \alpha}{\sum_{\bar{z}_2}{N_{z'_ 2,\bar{z}_2}} + K \alpha}, 
$$

<!--
ただし，\\( K \\) はGMMのクラス数である．
この確率を用いて遷移を考慮したそれぞれのクラスに分類される確率を計算し，GMMへ送信する．
GMMは，送られた確率も用いて再度分類を行うことでデータの遷移を考慮した分類が行われる．
-->

where \\( K \\) is the number of classes.
MM computes the probabilities that \\(z_2)\\ are classified into each class based on the transition probabilities, and sends them to GMM.
GMM classifies again using the received probabilities so that the classification is performed in consideration of the data transition.


<div align="center">
<img src="img/vae-gmm-mm/vae-gmm-mm.png" width="750px">
</div>

### Codes
Firstly, we import the necessary modules.

```
import serket as srk
import vae
import gmm
import mm
import numpy as np
```

Then, data and correct labels are loaded.
The data is sent as observations to the connected modules by using `srk.Observation`.

```
obs = srk.Observation( np.loadtxt( "data.txt" ) )
data_category = np.loadrxt( "category.txt" )
```

The modules VAE, GMM, and MM used in the integrated model are defined.
In the VAE, the dimensions of the latent variables are 18, the number of epochs is 200 and batch size is 500.
In the GMM, the data is classified into 10 classes, and optional argument `data_category` is correct labels and used to compute classification accuracy. 

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
mm1 = mm.MarkovModel()
```

The defined modules are connected and construct the integrated model.

```
vae1.connect( obs )  # connect obs to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
mm1.connect( gmm1 )  # connect gmm1 to mm1
```

Finally, the parameters of the whole model are learned by alternately updating the parameters of each module through exchanging messages. 

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
    mm1.update()  # train mm1
```

### Result
If training the model is succeeded, `module001_vae`, `module002_gmm`, and `module003_mm` directories are created.
The parameters of each module, probabilities, accuracy, and so on are stored in each directory.
The result and the accuracy of the classification are stored in `module002_gmm`.
The indexes of the classes into which each data is classified are saved in `class_learn.txt`, and the classification accuracy is saved in `acc_learn.txt`.
