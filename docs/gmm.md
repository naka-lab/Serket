---
layout: default
---
## GMM(Gaussian Mixture Model)

```
gmm.GMM( K, itr=100, name="gmm", category=None, mode="learn" )
```
### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of cluster |
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
学習に成功すると`module{i}_gmm`ディレクトリが作成される．  
ディレクトリ内には以下のファイルが保存される．
    - `model.pickle`: モデルパラメータが保存されている．
    - `acc_{mode}.txt`: categoryが与えられたとき計算された精度．
    - `class_{mode}.txt`: 分類したクラスz．
    - `mu_{mode}.txt`: データdが分類されたクラスzの分布の平均．
    - `Pdz_{mode}.txt`: データdがクラスzである確率．  
{mode}には選択したmode(learn or recog)が入る．