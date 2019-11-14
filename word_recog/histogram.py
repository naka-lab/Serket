# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import codecs
import os
import shutil
import numpy as np

from . import utils


# 単語ヒストグラムの作成
def make_histogram( wave_id, segmFile , saveDir):
    sentences = codecs.open( segmFile, "r" , "sjis" ).readlines()
    codebook = []

    sentences = [ s.replace("¥r","").replace("¥n","") for s in sentences ]

    # 保存フォルダ削除＆作成
    if os.path.exists( saveDir ):
        shutil.rmtree( saveDir )
    os.mkdir(saveDir)

    # コードブックを作成
    for sentence in sentences:
        for w in sentence.split():
            if not w in codebook:
                codebook.append( w )

    # 文章ごとのヒストグラムを作成
    dim = len(codebook)
    sentenceHsit = []

    for s in sentences:
        hist = [0,]*dim
        for w in s.split():
            i = codebook.index(w)
            hist[i] += 1
        sentenceHsit.append(hist)

    # 物体ごとのヒストグラム作成
    numObj = wave_id[-1][0]+1
    objectHist = [ [0,]*dim for _ in range(numObj) ]
    for n,s in enumerate(sentences):
        o = wave_id[n][0] 
        for w in s.split():
            i = codebook.index(w)
            objectHist[o][i] += 1

    np.savetxt( saveDir + "/sentencesHist.txt", sentenceHsit, fmt=str("%d") )
    np.savetxt( saveDir + "/objectHist.txt", objectHist, fmt=str("%d") )     
    utils.save_lines( codebook , saveDir + "/codebook.txt" )
    return objectHist, sentenceHsit


def main():
    # 音声認識の1bestのみでヒストグラム作成
    make_histogram( [[0,0,0], [1,0,0], [2,0,0]], "test.txt", "aa" )





if __name__ == '__main__':
    main()