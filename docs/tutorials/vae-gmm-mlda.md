---
layout: default
---
## VAE + GMM + MLDA
We construct a model integrating VAE, GMM, and MLDA and do unsupervised classification using multimodal information.

### Data
We use [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and [Spoken Arabic Digit Data Set](https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit).
The number of data is 3000.
MNIST dataset is handwritten digit image data.
Spoken Arabic Digit Data Set is data obtained by converting spoken Arabic digits into MFCC features and published in UCI Machine Learning Repository.
In this example, we use MFCC features converted to HAC features.
A detailed explanation of HAC features is [here](https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html)．

### Model
VAEは，観測 \\( \boldsymbol{o}_ 1 \\) をエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_ 1 \\) に圧縮し，GMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_ 1 \\) を分類し，\\( t \\) 番目のデータがクラス \\( z_ {2,t} \\) に分類される確率 \\( P(z_ {2,t} \mid \boldsymbol{z}_ {1,t}) \\) をMLDAへ送信，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
VAEは，\\( \boldsymbol{\mu} \\) を用いることでGMMの分類に適した潜在空間が学習する．
MLDAは，GMMから送られてきた確率を用いることで潜在変数 \\( z_ 2 \\) を観測として扱い，\\( z_ 2 \\) と観測 \\( \boldsymbol{o}_ 2 \\) を分類し，GMMへ確率 \\( P(z_ {2,t} \mid z_ {3,t}, \boldsymbol{o}_ {2,t}) \\)を送信する．
GMMは，送られてきた確率も用いて再度分類を行うことで，MLDAの影響を受け \\( z_ 3, \boldsymbol{o}_ 2 \\) を考慮した分類が行われる．

<div align="center">
<img src="img/vae-gmm-mlda/vae-gmm-mlda.png" width="620px">
</div>

### Codes
Firstly, we import the necessary modules.

```
import serket as srk
import vae
import gmm
import mlda
import numpy as np
```

Secondly, we load data and correct labels.
The data are sent as observations to the connected module by `srk.Observation`.

```
obs1 = srk.Observation( np.loadtxt( "data1.txt" ) )  # image data
obs2 = srk.Observation( np.loadtxt( "data2.txt" ) )  # audio data
data_category = np.loadrxt( "category.txt" )
```

Thirdly, we define each module.
We define VAE that compresses the data into 18 dimensions, whose epoch number is 200 and batch size is 500.
We define GMM that classifies the data into ten classes and give `data_category` as correct labels.
We define MLDA that classifies the data into ten classes and give `[200,200]` as the weight of each modality and  `data_category` as correct labels.

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
mlda1 = mlda.MLDA( 10, [200,200], category=data_category )
```

Fourthly, we connect the modules and construct the model.

```
vae1.connect( obs1 )  # connect obs1 to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
mlda1.connect( obs2, gmm1 )  # connect obs2 and gmm1 to mlda1
```

Finallly, we optimize the whole model by repeatedly updating the parameters of each module and exchanging messages．

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
    mlda1.update()  # train mlda1
```

### Result
If training the model is successful, `module002_vae`, `module003_gmm`, and `module004_mlda` directories are created.
The parameters of each module, probabilities, accuracy, and so on are stored in each directory.
The result and the accuracy of the classification are stored in `module004_mlda`.
The indexes of classes in which each data is classified are saved in `class_learn.txt`, and the classification accuracy is saved in `acc_learn.txt`.
