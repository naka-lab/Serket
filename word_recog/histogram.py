# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import codecs
import os
import shutil
import numpy as np

from . import utils


# 単語ヒストグラムの作成
def make_histogram( wave_id, segmFile, saveDir, threshold ):
    sentences = codecs.open( segmFile, "r" , "sjis" ).readlines()
#    codebook = []
    codebook_dict = {}

    sentences = [ s.replace("¥r","").replace("¥n","") for s in sentences ]

    # 保存フォルダ削除＆作成
    if os.path.exists( saveDir ):
        shutil.rmtree( saveDir )
    os.mkdir(saveDir)

    # コードブックを作成
    for sentence in sentences:
        for w in sentence.split():
#            if not w in codebook:
#                codebook.append( w )
            # codebookの作成・頻度の集計
            if not w in codebook_dict:
                codebook_dict[w] = 1
            else:
                codebook_dict[w] += 1
                
    # codebookと頻度に分解
    codebook = np.array( list( codebook_dict.keys() ) )
    code_freq = np.array( list( codebook_dict.values() ) )
    # 頻度の低い単語の削除
    codebook = np.delete( codebook, np.where( code_freq<threshold ) )

    # 文章ごとのヒストグラムを作成
    dim = len(codebook)
    sentenceHist = []

    for s in sentences:
        hist = [0,]*dim
        for w in s.split():
#            i = codebook.index(w)
#            hist[i] += 1
            if w in codebook:
                i = np.where( codebook==w )[0][0]
                hist[i] += 1
        sentenceHist.append(hist)

    # 物体ごとのヒストグラム作成
    numObj = wave_id[-1][0]+1
    objectHist = [ [0,]*dim for _ in range(numObj) ]
    for n,s in enumerate(sentences):
        o = wave_id[n][0] 
        for w in s.split():
#            i = codebook.index(w)
#            objectHist[o][i] += 1
            if w in codebook:
                i = np.where( codebook==w )[0][0]
                objectHist[o][i] += 1

    np.savetxt( saveDir + "/sentencesHist.txt", sentenceHist, fmt=str("%d") )
    np.savetxt( saveDir + "/objectHist.txt", objectHist, fmt=str("%d") )     
    utils.save_lines( codebook , saveDir + "/codebook.txt" )
    return objectHist, sentenceHist


def main():
    # 音声認識の1bestのみでヒストグラム作成
    make_histogram( [[0,0,0], [1,0,0], [2,0,0]], "test.txt", "aa" )


if __name__ == '__main__':
    main()