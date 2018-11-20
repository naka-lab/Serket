---
layout: default
---
## MLDA(Multimoda Latent Dirichlet Allocation)

```
mlda.MLDA( K, weights=None, itr=100, name="mlda", category=None, mode="learn" )
```

MLDAはLDAをマルチモーダル情報に拡張したトピックモデルであり，クラスタリングを行うモジュールである．
各データがそれぞれのクラスに分類される確率と各データからモダリティの特徴が発生する確率を計算し，接続されたモジュールへ送信する．


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of cluster |
| weghits   | array | Weight for each modality |
| itr       | int | Number of iteration |
| name      | string | Name of module |
| category  | array | Correct class label |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |


### Method

- .connect()  
観測またはモジュールと接続しモデルを構築する．
- .update()  
モデルパラメータを推定し確率などを計算する．  
"learn"モード時には学習を行い，"recog"モード時では未知データに対する予測を行う．  
学習に成功すると`module{i}_mlda`ディレクトリが作成される．  
ディレクトリ内には以下のファイルが保存される．（{mode}には選択したmode(learn or recog)が入る．）
    - `model.pickle`: モデルパラメータが保存されている．
    - `acc_{mode}.txt`: categoryが与えられたとき計算された精度．
    - `categories_{mode}.txt`: 分類したクラスz．
    - `Pdz_{mode}.txt`: データdがクラスzである確率．
    - `Pmdw[i]_{mode}.txt`: データdからモダリティiの特徴wが発生する確率．