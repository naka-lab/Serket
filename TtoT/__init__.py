#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import copy

class TtoT(srk.Module):
    def __init__( self, name="tt" ):
        super(TtoT, self).__init__(name, False)

    def update( self ):
        data = self.get_observations()
#        bias = self.get_backward_msg()
        
        K = len( data[0][0] )   # 潜在変数の次元数
        N = len( data[0] )      # データ数
        M = len( data )         # モジュール数
        changed = True          # 変化したかどうか
        
        # 確率が送られてきてないモジュールは一様分布を代入
        for m in range(M):
            if np.all(data[m]==None):
                data[m] = np.ones([N, K]) / K
        
        # 各モジュールの確率をワンホット化(インデックスをそろえる計算を楽にするため)
        onehot_data = []
        for m in range(M):
            # 確率の最も高いインデックスを取得
            max_idx = np.argmax(data[m], axis=1)
            # 確率をワンホット化
            onehot_data.append( np.eye(K)[max_idx] )
        
        # 各モジュールの確率のインデックスをそろえる
        for m in range(1,M):
            max_acc = 0  # 精度の最大値(更新のための基準)
            while changed:
                changed = False
                for i in range(K):
                    for j in range(K):
                        # 入れ替える用の一時的なデータの作成
                        tmp_onehot_data = copy.deepcopy( onehot_data[m] )
                        tmp_data = copy.deepcopy( data[m] )
        
                        # i列とj列を入れ替える
                        tmp_onehot_data[:, i], tmp_onehot_data[:, j] = onehot_data[m][:, j], onehot_data[m][:, i]
                        tmp_data[:, i], tmp_data[:, j] = data[m][:, j], data[m][:, i]
        
                        # モジュール[0]を基準として精度を計算
                        acc = ( np.all(tmp_onehot_data==onehot_data[0], axis=1) ).sum() / float(N)
        
                        # 精度が高くなっていれば保存
                        if acc > max_acc:
                            max_acc = acc
                            onehot_data[m] = tmp_onehot_data
                            data[m] = tmp_data
                            changed = True
        
        # スムージング(極端な確率を補正・正規化)
        for m in range(M):
            data[m] = data[m] + 0.01
            data[m] = ( data[m].T / data[m].sum(1) ).T
        
        # backward messageの計算
        back_msg = []
        for m in range(M):
            # m番目のモジュールの確率を取り除く
            tmp_data = np.delete( data, m, axis=0 )
            # m番目以外の確率の積を計算
            back_msg.append( np.prod(tmp_data, axis=0) )

        # メッセージの送信
#        self.set_forward_msg( Pdz )
        self.send_backward_msgs( back_msg )
