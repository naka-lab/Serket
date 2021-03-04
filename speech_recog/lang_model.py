# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import os
import codecs
import sys

# 言語モデルを作成するためのプログラムの場所
__file_path = os.path.dirname(os.path.abspath(__file__))
__make_lm = os.path.join( __file_path ,"MakeLM/MakeLM.bat" )

# filesで渡された複数ファイルを結合
def combine_files( files , dstfile ):
    fDst = codecs.open( dstfile , "w" , "sjis" )
    for file in files:
        lines = codecs.open( file , "r" , "sjis" ).readlines()
        for line in lines:
            line = line.replace("\r" , "").replace("\n" , "")
            if len(line) == 0:
                continue
            fDst.write( line )
            fDst.write( "\n" )
    fDst.close()

# 言語モデルの作成
def makelm( sentencefile , lmBaseName ):
    if os.path.exists( lmBaseName + ".bingram" ):
        os.remove( lmBaseName + ".bingram" )

    if os.path.exists( lmBaseName + ".htkdic" ):
        os.remove( lmBaseName + ".htkdic" )

    os.system( __make_lm + " " + sentencefile + " " + lmBaseName )

    # 正しく作成されたかチェック
    if not os.path.exists( lmBaseName + ".bingram" ):
        print( "言語モデルの作成に失敗！！" )
        sys.exit(0)


def main():
    makelm( "module000_speech_recog/init/recogres.seg.txt" , "test" )

if __name__ == '__main__':
    main()