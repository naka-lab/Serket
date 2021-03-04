#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import gmm
import serket as srk
import numpy as np
import os

class GMM(srk.Module):
    def __init__( self, K, itr=100, name="gmm", category=None, load_dir=None ):
        super(GMM, self).__init__(name, True)
        self.__K = K
        self.__itr = itr
        self.__category = category
        self.__load_dir = load_dir
        self.__n = 0

    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        N = len( data[0] )  # データ数

        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K

        data[0] = np.array( data[0], dtype=np.float32 )

        if self.__load_dir is None:
            save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        else:
            save_dir = os.path.join( self.get_name(), "recog" )

        # GMM学習
        Pdz, mu = gmm.train( data[0], self.__K, self.__itr, save_dir, Pdz, self.__category, self.__load_dir )
        
        self.__n += 1

        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( [mu] )
