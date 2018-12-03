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
VAEは，このようにして圧縮された潜在変数 \\( \boldsymbol{z}_1 \\) を接続されたモジュールへ送信する．
GMMは，VAEから送られてきた潜在変数 \\( \boldsymbol{z}_1 \\) を分類し，分類されたクラスの平均 \\( \boldsymbol{\mu} \\) をVAEへ送信する．
通常VAEの変分下限は次式で表される．

$$
\mathcal{L}( \boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{o} ) = -D_{KL} ( q_{ \boldsymbol{\phi} }( \boldsymbol{z}_1 \mid \boldsymbol{o} ) \| \mathcal{N} ( 0, \boldsymbol{I} ) ) + \mathbb{E}_{ q_{ \boldsymbol{\phi} }( \boldsymbol{z}_1 \mid  \boldsymbol{o} ) } [ \log{ p_{ \boldsymbol{\theta} } ( \boldsymbol{o} \mid \boldsymbol{z}_1 ) } ]
$$

Serketでは，GMMでの分類の影響を受けるため，データが分類されたクラスの平均 \\( \mu \\) を用いて変分下限を以下のように定義する．

$$
\mathcal{L}( \boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{o} ) = - \alpha D_{KL} ( q_{ \boldsymbol{\phi} } ( \boldsymbol{z}_1 \mid \boldsymbol{o} ) \| \mathcal{N} ( \boldsymbol{\mu}, \boldsymbol{I} ) ) + \mathbb{E}_{ q_{ \boldsymbol{\phi} } ( \boldsymbol{z}_1 \mid \boldsymbol{o} ) } [ \log{ p_{ \boldsymbol{\theta} } ( \boldsymbol{o} \mid \boldsymbol{z}_1 ) } ]
$$

ただし， \\( D_{KL} \\) はKLダイバージェンスを表しており，\\( \alpha \\) はKLダイバージェンスの重みである.
これにより，GMMによって同じクラスに分類されたデータの潜在変数 \\( \boldsymbol{z}_1 \\) は似た値を持つこととなり，分類に適した潜在空間が学習される．

<div align="center">
<img src="img/vae-gmm/vae-gmm.png" width="530px">
</div>

### Codes
必要なモジュールをimportする．

```
import serket as srk
import vae
import gmm
import numpy as np
```

データと正解ラベルの読み込む．
`srk.Observation`により読み込んだデータを接続されたモジュールに観測として送信する．

```
obs = srk.Observation( np.loadtxt( "data.txt" ) )
data_category = np.loadtxt( "category.txt" )
```

各モジュールを定義する．
VAEは圧縮後の次元を18次元，エポック数を200，バッチサイズを500として定義する．
GMMはクラス数を10，正解ラベルとしてdata_categoryを与えて定義する．

```
vae1 = vae.VAE( 18, itr=200, batch_size=500 )
gmm1 = gmm.GMM( 10, category=data_category )
```

モジュールを接続し，モデルを構築する．

```
vae1.connect( obs )   # connect obs to vae1
gmm1.connect( vae1 )  # connect vae1 to gmm1
```

各モジュールのパラメータの更新とメッセージのやり取りを繰り返し行うことでモデル全体の最適化を行う．

```
for i in range(5):
    vae1.update()  # train vae1
    gmm1.update()  # train gmm1
```

### Result
モデルの学習が成功すると`module001_vae`と`module002_gmm`ディレクトリが作成される．
それぞれのディレクトリには，モデルのパラメータや確率，精度などが保存されている．
圧縮された潜在変数 \\( z_1 \\) は`module001_vae`内の`z_learn.txt`に保存されている．
潜在変数 \\( z_1 \\) を主成分分析により2次元に圧縮しプロットしたグラフを以下に示す．

<div align="center">
<img src="img/vae-gmm/pca.png" width="550px">
</div>

最適化前では同じクラスであるデータ点が空間上に広く分散しているのに対して，最適化後ではクラスごとに似た値を持ってまとまっている．
メッセージのやり取りによって分類に適した潜在空間が学習されていることが確認できる．
分類の結果や精度は`module002_gmm`内に保存されており，`class_learn.txt`に各データが分類されたクラスのインデックス，`acc_learn.txt`に分類の精度が保存されている．
