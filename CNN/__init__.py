#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import cv2
import os


class CNNFeatureExtractor(srk.Module):
    def __init__(self, fileames, name="CNNFeatureExtracter" ):
        super(CNNFeatureExtractor, self).__init__(name, False)
        self.is_ninitilized = False
        self.features = []

        proto_file = os.path.join( os.path.dirname(__file__), "bvlc_googlenet.prototxt" )
        caffemodel_file = os.path.join( os.path.dirname(__file__), "bvlc_googlenet.caffemodel" )
        self.net = cv2.dnn.readNetFromCaffe( proto_file, caffemodel_file)


        if fileames!=None:
            self.filenames = fileames

            for fname in self.filenames:        
                # 必要なファイルを読み込む
                image = cv2.imread( fname )
                
                # 認識処理
                blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
                self.net.setInput(blob)
                preds = net.forward("pool5/7x7_s1")
                
                f = preds[0, :, 0, 0]
                self.features.append(f)
            self.is_ninitilized = True
        
            self.set_forward_msg( self.features )

    def update(self):
        data = self.get_observations()[0]

        # 新たなデータがあれば特徴抽出
        n_feat = len(self.features)
        n_data = len(data)
        if n_data>n_feat:
            for i in range( n_feat, n_data ):
                blob = cv2.dnn.blobFromImage(data[i], 1, (224, 224), (104, 117, 123))
                self.net.setInput(blob)
                preds = self.net.forward("pool5/7x7_s1")
                
                f = preds[0, :, 0, 0]
                self.features.append(f)

        self.set_forward_msg( self.features )
 
