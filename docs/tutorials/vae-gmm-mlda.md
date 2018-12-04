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
VAEは，観測 \\(\boldsymbol{o}_1\\) をエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\(\boldsymbol{z}_1\\) に圧縮し，GMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\(\boldsymbol{z}_1\\) を分類し，\\(t\\) 番目のデータがクラス \\(z_{2,t}\\) に分類される確率 \\(P(z_ {2,t} \mid \boldsymbol{z}_{1,t})\\) をMLDAへ送信，分類されたクラスの平均 \\(\boldsymbol{\mu}\\) をVAEへ送信する．
VAEは，\\(\boldsymbol{\mu}\\) を用いることでGMMの分類に適した潜在空間が学習される．
MLDAは，GMMから送られてきた確率を用いることで潜在変数 \\(z_2\\) を観測として扱い，\\(z_2\\) と \\(\boldsymbol{o}_2\\) を分類し，GMMへ確率 \\(P(z_ {2,t} \mid z_ {3,t}, \boldsymbol{o}_ {2,t})\\)を送信する．
GMMは，送られてきた確率も用いて再度分類を行うことで，MLDAの影響を受け \\(z_3, \boldsymbol{o}_2\\) を考慮した分類が行われる．

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
分類の結果や精度は`module004_mlda`内に保存されており，`categories_learn.txt`に各データが分類されたクラスのインデックス，`acc_learn.txt`に分類の精度が保存されている．
