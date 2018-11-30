---
layout: default
---
## VAE + GMM + MM
GMMとMMを組み合わせることでHMM (Hidden Markov Model)を構築することができる．
VAEによる次元圧縮とHMMによる推移を考慮した教師なし分類の相互学習を行う．

### Data
MNISTデータセットを使用する．
データ数は3000であり，推移を学習するため0,1,2,3,4,5,6,7,8,9,0,\\( \cdots \\) のように並び替えたものを使用する．

### Model
VAEで，観測 \\( \boldsymbol{o} \\) がエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_1 \\)に圧縮される．
VAEは，圧縮された潜在変数 \\( \boldsymbol{z}_1 \\) をGMMへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_1 \\) を分類し，確率などを計算する．
GMMは，確率 \\( P(\boldsymbol{z}_2|\boldsymbol{z}_1) \\) をMMモジュールへ送信し，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
MMは，送られてきた確率 \\( P(\boldsymbol{z}_2|\boldsymbol{z}_1) \\)を用いて繰り返しサンプリングを行い，次のように遷移回数をカウントする．

$$
z'_ 2 \sim P(z_{2,t}|\boldsymbol{z}_{1,t})\\
z_2 \sim P(z_{2,t+1}|\boldsymbol{z}_{1,t+1})\\
N_{z'_ 2,z_2}++
$$

ここで，\\( P(z_{2,t}|\boldsymbol{z}_{1,t}) \\) は \\(　t　\\)　番目のデータがクラス \\( z_{2,t} \\) に分類される確率である．
この値から遷移確率 \\( P(z_2|z'_ 2) \\) は次のように計算することができる．

$$
P(z_2|z'_ 2) = \frac{N_{z'_ 2,z_2} + \alpha}{\sum_{\bar{z}_2}{N_{z'_ 2,\bar{z}_2}} + K \alpha}
$$

ただし，\\( K \\) はGMMのクラス数である．
この確率を用いて遷移を考慮したそれぞれのクラスに分類される確率を計算し，GMMへ送信する．
GMMは，送られた確率も用いて再び分類を行う．
すなわち，この確率を利用することでデータの遷移を考慮した分類が行われる．

### Codes
必要なモジュールをimportする．

```
import Serket as srk
import vae
import gmm
import mm
import numpy as np
```

データと正解ラベルを読み込む．

```
obs = srk.Observation( np.loadtxt( "data.txt" ) )
data_category = np.loadrxt( "category.txt" )
```

モジュールを定義する．
VAEはエポック数を200，バッチサイズを500として784次元のデータを18次元に圧縮する．
GMMはクラス数を10とし，正解ラベルを与える．

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
mm1 = mm.MarkovModel()
```

モジュールを接続し，モデルを構築する．

```
vae1.connect( obs )
gmm1.connect( vae1 )
mm1.connect( gmm1 )
```

パラメータの更新を繰り返し行い，最適化を行う．

```
for i in range(5):
    vae1.update()
    gmm1.update()
    mm1.update()
```
