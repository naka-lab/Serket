#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
from janome.tokenizer import Tokenizer
import os


class BoW(srk.Module):
    def __init__(self, sentences=None, name="Bow" ):
        super(BoW, self).__init__(name, False)
        self.__word_dict = []
        self.__tokenizer = Tokenizer()
        self.__histograms = []
        self.__itr = 0

        if sentences!=None:
            self.__histograms = self.tokenize( sentences ) 
        
            self.set_forward_msg( self.__histograms )
            self.save()

    def tokenize(self, sentences):
        histograms = []
        for s in sentences:
            hist = [ 0 for _ in range(len(self.__word_dict)) ]

            if sys.version_info.major==2:
                s = s.decode("utf8")

            words = self.__tokenizer.tokenize( s, wakati=True )

            for w in words:
                if w in self.__word_dict:
                    idx = self.__word_dict.index( w )
                    hist[idx] += 1
                else:
                    self.__word_dict.append(w)
                    hist.append(1)
            histograms.append( hist )
        # 0パディング
        dim = len(self.__word_dict)
        histograms  = np.array([ hist+[0]*(dim-len(hist)) for hist in histograms ], dtype=int)

        return histograms


    def save(self):
        if not os.path.exists( self.get_name() ):
            os.mkdir( self.get_name() )

        save_dir = os.path.join( self.get_name(), "%03d" % self.__itr )
        if not os.path.exists( save_dir ):
            os.mkdir( save_dir )

        path = os.path.join( save_dir, "histograms.txt" )
        np.savetxt( str(path), self.__histograms, fmt=str("%d") )

        path = os.path.join( save_dir, "dict.txt" )
        with open(path, "w") as f:
            if sys.version_info.major==2:
                f.write( "\n".join(self.__word_dict).encode("utf8") )           
            else:
                f.write( "\n".join(self.__word_dict) )

        self.__itr += 1

    def update(self):
        data = self.get_observations()[0]

        # 新たなデータがあれば特徴抽出
        if len(data)>0:
            self.__histograms = self.tokenize( data )

        self.set_forward_msg( self.__histograms )
        self.save()
 
