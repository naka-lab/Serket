---
layout: default
---
## VAE + GMM + MLDA
VAE, GMM, MLDAを統合したマルチモーダル情報を用いた数字の分類を行う．

### Data
MNISTデータセットおよび Spoken Arabic Digit Data Set（データ数：3000）を使用する．
Spoken Arabic Digit Data Set は UCI Machine Learning Repository にて公開されている数字発話をMFCC特徴量に変換したデータである．
このMFCC特徴量をHAC特徴量に変換したものを使用する．
HAC特徴量の詳しい説明は[こちら](https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html)．

### Model
VAEで，観測 \\( \boldsymbol{o}_1 \\) がエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_1 \\)に圧縮される．
VAEは，圧縮された潜在変数 \\( \boldsymbol{z}_1 \\) をGMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_1 \\) から確率などを計算し，分類を行う．
GMMは確率 \\( P(z_2 \mid \boldsymbol{z}_1) \\) をMMへ送信し，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
MLDAは，GMMから送られてきた確率を用いて\\( \hat{z}_2 \\) を観測変数として扱うことができる．

$$
\hat{z}_2 \sim  P(z_2 \mid \boldsymbol{z}_1)
$$

したがって，確率を計算し分類することができる．
そして，GMMへ確率 \\( P(z_2 \mid z_3, \boldsymbol{o}_2) \\)を送信する．


<div align="center">
<img src="img/vae-gmm-mlda/vae-gmm-mlda.png" width="450px">
</div>

### Codes
必要なモジュールをimportする．

```
import serket as srk
import vae
import gmm
import mlda
import numpy as np
```

データと正解ラベルを読み込む．
`srk.Observation`により読み込んだデータを接続されたモジュールに観測として送信する．

```
obs1 = srk.Observation( np.loadtxt( "data1.txt" ) )  # image data
obs2 = srk.Observation( np.loadtxt( "data2.txt" ) )  # audio data
data_category = np.loadrxt( "category.txt" )
```

各モジュールを定義する．
VAEは圧縮後の次元を18次元，エポック数を200，バッチサイズを500として定義する．
GMMはクラス数を10，正解ラベルとしてdata_categoryを与えて定義する．
MLDAはクラス数を10，各モダリティの重みをそれぞれ200，正解ラベルとしてdata_categoryを与えて定義する．

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
mlda1 = mlda.MLDA( 10, [200,200], category=data_category )
```

モジュールを接続し，モデルを構築する．

```
vae1.connect( obs1 )  # connect obs1 to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
mlda1.connect( obs2, gmm1 )  # connect obs2 and gmm1 to mlda1
```

各モジュールのパラメータの更新とメッセージのやり取りを繰り返し行うことでモデル全体の最適化を行う．

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
    mlda1.update()  # train mlda1
```

### Result
モデルの学習が成功すると`module002_vae`，`module003_gmm`，`module004_mlda`ディレクトリが作成される．
それぞれのディレクトリには，モデルのパラメータや確率，精度などが保存されている．
分類の結果や精度は`module004_gmm`内に保存されており，`categories_learn.txt`に各データが分類されたクラスのインデックス，`acc_learn.txt`に分類の精度が保存されている．
