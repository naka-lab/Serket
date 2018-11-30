---
layout: default
---
## VAE + GMM
VAEによる次元圧縮とGMMによる教師なし分類の相互学習を行う．

### Data
MNISTデータセット（データ数：3000）を使用する．

### Model
VAEでは，観測 \\( \boldsymbol{o} \\) がエンコーダーにあたるニューラルネットを通して任意の次元の潜在変数 \\( \boldsymbol{z}_1 \\) に圧縮される．
そして，潜在変数 \\( \boldsymbol{z}_1 \\) がデコーダーにあたるニューラルネットを通して元の次元に復元され，その値と観測 \\( \boldsymbol{o} \\) 同じになるように学習される．
VAEモジュールは，このようにして圧縮された潜在変数 \\( \boldsymbol{z}_1 \\) を接続されたモジュールへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_1 \\) を分類し，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
通常VAEの変分下限は次式で表される．

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\boldsymbol{o})=-D_{KL}(q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{o})||\mathcal{N}(0,\boldsymbol{I}))+\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{o})}[\log{p_{\boldsymbol{\theta}}(\boldsymbol{o}|\boldsymbol{z}_1)}]
$$

Serketでは，GMMでのクラスタリングの影響を受けるため，データが分類されたクラスの平均 \\( \mu \\) を用いて変分下限を以下のように定義する．

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\boldsymbol{o})=-D_{KL}(q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{o})||\mathcal{N}(\boldsymbol{\mu},\boldsymbol{I}))+\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{o})}[\log{p_{\boldsymbol{\theta}}(\boldsymbol{o}|\boldsymbol{z}_1)}]
$$

これにより，GMMによって同じクラスに分類されたデータの潜在変数 \\( \boldsymbol{z}_1 \\) は似た値を持つこととなり，クラスタリングに適した潜在空間が学習される．

### Codes
必要なモジュールをimportする．

```
import serket as srk
import vae
import gmm
import numpy as np
```

観測と正解ラベルの読み込む．

```
data = np.loadtxt( "data.txt" )
data_category = np.loadtxt( "category.txt" )
```

モジュールを定義する．
VAEはエポック数を200，バッチサイズを500として784次元のデータを18次元に圧縮する．
GMMはクラス数を10とし，正解ラベルを与える．

```
obs = srk.Observation( data )
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
```

モジュールを接続し，モデルを構築する．

```
vae1.connect( obs )
gmm1.connect( vae1 )
```

パラメータの更新を繰り返し行い，最適化を行う．

```
for i in range(5):
    vae1.update()
    gmm1.update()
```
