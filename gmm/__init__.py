#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import gmm
import serket as srk
import numpy as np


class GMM(srk.Module):
    def __init__(self, K, itr=100, name="gmm", category=None ):
        super(GMM, self).__init__(name, True)
        self.__K = K
        self.__itr = itr
        self.__category = category

    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        N = len( data[0] )  # データ数

        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K

        data[0] = np.array( data[0], dtype=np.float32 )

        # GMM学習
        Pdz, mu = gmm.train( data[0], self.__K, self.__itr, self.get_name(), bias_dz=Pdz, categories=self.__category )

        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( [mu] )
