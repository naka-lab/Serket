#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )


import serket as srk
import numpy as np
import os

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from . import mlda

class MLDA(srk.Module):
    def __init__( self, K, weights=None, itr=100, name="mlda", category=None, load_dir=None ):
        super(MLDA, self).__init__(name, True)
        self.__K = K
        self.__weights = weights
        self.__itr = itr
        self.__category = category
        self.__load_dir = load_dir
        self.__n = 0
        
    def update(self, load_trained_model=None):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        M = len( data )     # モダリティ数
        
        for m in range(M):
            if data[m].all() is not None:
                N = len( data[m] )     # データ数
                break
        
        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K
    
        # データの正規化処理
        for m in range(M):
            if data[m].all() is not None:
                data[m][ data[m]<0 ] = 0
            
        if self.__weights is not None:
            for m in range(M):
                if data[m].all() is not None:
                    divider = np.where( data[m].sum(1)==0, 1, data[m].sum(1) )
                    data[m] = ( data[m].T / divider ).T * self.__weights[m]
        
        for m in range(M):
            if data[m].all() is not None:
                data[m] = np.array( data[m], dtype=np.int32 )
        
        if load_trained_model is None:
            save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        else:
            save_dir = os.path.join( self.get_name(), "%03drecog" % self.__n )
        
        # MLDA学習
        Pdz, Pdw = mlda.train( data, self.__K, self.__itr, save_dir, Pdz, self.__category, load_trained_model )
        
        self.__n += 1
        
        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( Pdw )