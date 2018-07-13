#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import mlda
import serket as srk
import numpy as np


class MLDA(srk.Module):
    def __init__(self, K, weights=None, itr=100, name="mlda", category=None ):
        super(MLDA, self).__init__(name, True)
        self.__K = K
        self.__weights = weights
        self.__itr = itr
        self.__category = category
        
    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        M = len( data )     # モダリティ数
        N = len( data[0] )  # データ数
        
        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K
   
    
        # データの正規化処理
        for m in range(M):     
            data[m][ data[m]<0 ] = 0
            
        if self.__weights!=None:
            for m in range(M):
                data[m] = (data[m].T / data[m].sum(1)).T * self.__weights[m]
        
        for m in range(M):
            data[m] = np.array( data[m], dtype=np.int32 )
        
        
        # MLDA学習
        Pdz, Pdw = mlda.train( data, self.__K, self.__itr, self.get_name(), bias_dz=Pdz, categories=self.__category )
        
        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( Pdw )