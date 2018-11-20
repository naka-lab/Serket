---
layout: default
---
## MM(Markov Model)

```
mm.MM( num_samp=100, name="mm", mode="learn" )
```
### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| num_samp  | int | Number of sampling |
| name      | string | Name of module |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |

### Method

- .connect()  
モジュールと接続しモデルを構築する．
- .update()  
モデルパラメータを推定し確率などを計算する．  
"learn"モード時には学習を行い，"recog"モード時では未知データに対する予測を行う．  
学習に成功すると`module{i}_mm`ディレクトリが作成される．  
ディレクトリ内には以下のファイルが保存される．
    - `model.pickle`: モデルパラメータが保存されている．
    - `msg_{mode}.txt`: データdがクラスzであるか確率．
    - `trans_prob_learn.txt`: 学習時に計算した遷移確率．  
{mode}には選択したmode(learn or recog)が入る．