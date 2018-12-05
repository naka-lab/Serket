---
layout: default
---
## VAE + GMM + MM
HMM (Hidden Markov Model) can be constructed by combining GMM and MM.
We construct a model integrating VAE, GMM, and MM, and do mutual learning of dimension compression by VAE and unsupervised classification considering transition by HMM.

### Data
We use [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
The number of data is 3000.
MNIST dataset is handwritten digit image data.
In order to learn the transition, use sorted like 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, \\( \cdots \\).

### Model
VAEは，観測 \\( \boldsymbol{o} \\) をエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_ 1 \\) に圧縮し，GMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_ 1 \\) を分類し，\\( t \\) 番目のデータがクラス \\( z_ {2,t} \\) に分類される確率 \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) をMMへ送信，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
VAEは，\\( \boldsymbol{\mu} \\) を用いることでGMMの分類に適した潜在空間が学習される．
MMは，送られてきた確率 \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) を用いて繰り返しサンプリングを行い，次のように遷移回数をカウントする．

$$
z'_ 2 \sim P(z_{2,t} \mid \boldsymbol{z}_{1,t})\\
z_2 \sim P(z_{2,t+1} \mid \boldsymbol{z}_{1,t+1})\\
N_{z'_ 2,z_2}++
$$

この値から遷移確率 \\( P(z_ 2 \mid z'_ 2) \\) は次のように計算することができる．

$$
P(z_2 \mid z'_ 2) = \frac{N_{z'_ 2,z_2} + \alpha}{\sum_{\bar{z}_2}{N_{z'_ 2,\bar{z}_2}} + K \alpha}
$$

ただし，\\( K \\) はGMMのクラス数である．
この確率を用いて遷移を考慮したそれぞれのクラスに分類される確率を計算し，GMMへ送信する．
GMMは，送られた確率も用いて再度分類を行うことでデータの遷移を考慮した分類が行われる．

<div align="center">
<img src="img/vae-gmm-mm/vae-gmm-mm.png" width="450px">
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

Secondly, we load data and correct labels.
The data are sent as observations to the connected module by `srk.Observation`.

```
obs = srk.Observation( np.loadtxt( "data.txt" ) )
data_category = np.loadrxt( "category.txt" )
```

Thirdly, we define each module.
We define VAE that compresses to 18 dimensions, whose epoch number is 200 and batch size is 500.
We define GMM that classifies the data into ten classes and give `data_category` as correct labels.
We define MM without giving arguments.

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
mm1 = mm.MarkovModel()
```

Fourthly, we connect modules and construct the model.

```
vae1.connect( obs )  # connect obs to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
mm1.connect( gmm1 )  # connect gmm1 to mm1
```

Finallly, we optimize the whole model by repeatedly updating the parameters of each module and exchanging messages.

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
    mm1.update()  # train mm1
```

### Result
If training the model is successful, `module001_vae`, `module002_gmm`, and `module003_mm` directories are created.
The parameters of each module, probabilities, accuracy, and so on are stored in each directory.
The result and the accuracy of the classification are stored in `module002_gmm`.
The indexes of classes in which each data is classified are saved in `class_learn.txt`, and the classification accuracy is saved in `acc_learn.txt`.
