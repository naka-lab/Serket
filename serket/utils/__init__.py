#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import serket as srk

# 同じグループデータで和をとって，1グループにつき一つのベクトルに変換
class GroupSum(srk.Module):
    def __init__( self, group_idx, name="SumGroup" ):
        super(GroupSum, self).__init__( name, True)
        self.__group_idx = np.array(group_idx)

    def update(self):
        # グループごとに和をとってまとめる
        data = self.get_observations()
        num_episodes = np.max( self.__group_idx )+1

        fwd_msg = [ np.sum( data[0][self.__group_idx==e, :], axis=0 ) for e in range(num_episodes)]
        fwd_msg = np.array(fwd_msg)

        self.set_forward_msg( fwd_msg )

        # グループごとに計算された確率をデータ点ごとに複製
        prob = self.get_backward_msg() 

        if not prob is None:
            bk_msg = [[]]
            for e in self.__group_idx:
                bk_msg[0].append( prob[e] )
            bk_msg = np.array(bk_msg)

            self.send_backward_msgs( bk_msg )


class TextSaver(srk.Module):
    def __init__( self, filename, name="TextSaver" ):
        super(TextSaver, self).__init__( name, True)
        self.__filename = filename

    def update(self):
        data = self.get_observations()
        self.set_forward_msg( data[0] )
        np.savetxt( self.__filename, data[0] )

        prob = self.get_backward_msg() 
        if not prob is None:
            self.send_backward_msgs( prob )


class Buffer(srk.Module):
    def __init__( self, size=-1, name="Buffer" ):
        super(Buffer, self).__init__( name, True)
        self.__size = size
        self.__data = []

    def update(self):
        new_data = self.get_observations()
        N = len(new_data[0])

        for n in range(N):
            self.__data.append( new_data[0][n] )

            if self.__size>0 and len(self.__data)>self.__size:
                self.__data.pop(0)

        # 0パディング
        dim = np.max([ len(d) for d in self.__data] )
        self.__data  = [ list(d)+[0]*(dim-len(d)) for d in self.__data ]

        self.set_forward_msg( np.array( self.__data ) )


