#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import cv2

class CNNFeatureExtractor(srk.Module):
    def __init__(self, fileames, name="CNNFeatureExtracter" ):
        super(CNNFeatureExtractor, self).__init__(name, False)
        self.is_ninitilized = False
        self.filenames = fileames
        
        proto_file = "bvlc_googlenet.prototxt"
        caffemodel_file = "bvlc_googlenet.caffemodel"
        net = cv2.dnn.readNetFromCaffe( proto_file, caffemodel_file)
        
        features = []

        for fname in self.filenames:        
            # 必要なファイルを読み込む
            image = cv2.imread( fname )
            
            # 認識処理
            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            preds = net.forward("pool5/7x7_s1")
            
            f = preds[0, :, 0, 0]
            features.append(f)
        self.is_ninitilized = True
    
        self.set_forward_msg( features )
