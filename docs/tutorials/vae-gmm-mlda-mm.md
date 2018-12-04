---
layout: default
---
## VAE + GMM + MLDA + MM
MLDAとMMを組み合わせることでHMM (Hidden Markov Model)を構築することができる．
VAE, GMM, MLDA, MMを統合することで遷移も考慮したマルチモーダル情報を用いた数字の分類を行う．

### Data
MNISTデータセットおよび Spoken Arabic Digit Data Set（データ数：3000）を使用する．
Spoken Arabic Digit Data Set は UCI Machine Learning Repository にて公開されている数字発話をMFCC特徴量に変換したデータである．
このMFCC特徴量をHAC特徴量に変換したものを使用する．
HAC特徴量の詳しい説明は[こちら](https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html)．
推移を学習するため0,1,2,3,4,5,6,7,8,9,0,\\( \cdots \\) のように並び替えたものを使用する．

### Model
VAEは，観測 \\(\boldsymbol{o}_1\\) をエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\(\boldsymbol{z}_1\\)に圧縮し，GMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\(\boldsymbol{z}_1\\) を分類し，\\(t\\) 番目のデータがクラス \\(z_{2,t}\\) に分類される確率 \\(P(z_{2,t} \mid \boldsymbol{z}_{1,t})\\) をMLDAへ送信，分類されたクラスの平均 \\(\boldsymbol{\mu}\\) をVAEへ送信する．
VAEは，\\(\boldsymbol{\mu}\\) を用いることでGMMの分類に適した潜在空間が学習される．
MLDAは，GMMから送られてきた確率を用いて潜在変数 \\(z_2\\) を観測として扱い，観測 \\(z_2\\) と \\(\boldsymbol{o}_2\\) を分類し，確率 \\(P(z_{3,t} \mid z_{2,t}, \boldsymbol{o}_{2,t})\\) をMMへ送信，確率 \\(P(z_{2,t} \mid z_{3,t}, \boldsymbol{o}_{2,t})\\) をGMMへ送信する．
GMMは，送られてきた確率も用いて再度分類を行うことで，MLDAの影響を受け \\(z_3, \boldsymbol{o}_2\\) を考慮した分類が行われる．
MMは，送られてきた確率 \\(P(z_{3,t} \mid z_{2,t}, \boldsymbol{o}_{2,t})\\) を用いて繰り返しサンプリングを行い，次のように遷移回数をカウントする．

$$
z'_ 3 \sim P(z_{3,t} \mid z_ {2,t}, \boldsymbol{o}_{2,t})\\
z_3 \sim P(z_{3,t+1} \mid z_ {2,t+1}, \boldsymbol{o}_{2,t+1})\\
N_{z'_ 3,z_3}++
$$

この値から遷移確率 \\(P(z_3 \mid z'_ 3)\\) は次のように計算することができる．

$$
P(z_3 \mid z'_ 3) = \frac{N_{z'_ 3,z_3} + \alpha}{\sum_{\bar{z}_3}{N_{z'_ 3,\bar{z}_3}} + K \alpha}
$$

ただし，\\(K\\) はMLDAのクラス数である．
この確率を用いて遷移を考慮したそれぞれのクラスに分類される確率を計算し，MLDAへ送信する．
MLDAは，送られた確率も用いて再度分類を行うことでデータの遷移を考慮した分類が行われる．

<div align="center">
<img src="img/vae-gmm-mlda-mm/vae-gmm-mlda-mm.png" width="680px">
</div>

### Codes
必要なモジュールをimportする．

```
import serket as srk
import vae
import gmm
import mlda
import mm
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
mm1 = mm.MarkovModel()
```

モジュールを接続し，モデルを構築する．

```
vae1.connect( obs1 )  # connect obs1 to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
mlda1.connect( obs2, gmm1 )  # connect obs2 and gmm1 to mlda1
mm1.connect( mlda1 ) # connect mlda1 to mm1
```

各モジュールのパラメータの更新とメッセージのやり取りを繰り返し行うことでモデル全体の最適化を行う．

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
    mlda1.update()  # train mlda1
    mm1.update()  # train mm1
```

### Result
モデルの学習が成功すると`module002_vae`，`module003_gmm`，`module004_mlda`，`module005_mm`ディレクトリが作成される．
それぞれのディレクトリには，モデルのパラメータや確率，精度などが保存されている．
分類の結果や精度は`module004_mlda`内に保存されており，`categories_learn.txt`に各データが分類されたクラスのインデックス，`acc_learn.txt`に分類の精度が保存されている．
