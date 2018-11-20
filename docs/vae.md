---
layout: default
---
## VAE(Variational AutoEncoder)

```
vae.VAE( latent_dim, weight_stddev=0.1, itr=5000, name="vae", hidden_encoder_dim=100,
      　　  hidden_decoder_dim=100, batch_size=None, KL_param=1, mode="learn"  )
```

次元圧縮を行うモジュールである．
通常のVAEの変分下限は以下であるが，

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta,&space;\phi;&space;o)&space;=&space;-D_{KL}(q_{\phi}(z_1|o)||\mathcal{N}(0,&space;I))&plus;\mathbb{E}_{q_{\phi}(z_1|o)}[\log&space;p_{\theta}(o|z_1)]" />

Serketでは以下のように定義することで接続されたモジュールから受信するメッセージ($\mu$)を用いて最適化される．

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(\theta,&space;\phi;&space;o)&space;=&space;-D_{KL}(q_{\phi}(z_1|o)||\mathcal{N}(\mu,&space;I))&plus;\mathbb{E}_{q_{\phi}(z_1|o)}[\log&space;p_{\theta}(o|z_1)]" />


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| latent_dim | int | Number of latent dimension |
| weight_stddev | float | Standard deviation of weight |
| itr       | int | Number of iteration |
| name      | string | Name of module |
| hidden_encoder_dim | int | Number of node in encoder |
| hidden_decoder_dim | int | Number of node in decoder |
| batch_size | int | Number of batch size |
| KL_param  | float | Weight for KL divergence |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |


### Method

- .connect()  
観測またはモジュールと接続しモデルを構築する．
- .update()  
モデルパラメータを推定し潜在変数などを計算する．  
"learn"モード時には学習を行い，"recog"モード時では未知データに対する予測を行う．  
学習に成功すると`module{i}_vae`ディレクトリが作成される．  
ディレクトリ内には以下のファイルが保存される．（{mode}には選択したmode(learn or recog)が入る．）
    - `model.ckpt`: モデルパラメータが保存されている．
    - `loss.txt`: 学習時のloss．
    - `x_hat_{mode}.txt`: decoderから出力された復元データ．
    - `z_{mode}.txt`: encoderにより圧縮された潜在変数．  