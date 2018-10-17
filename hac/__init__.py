#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import hac
import serket as srk
import os
import numpy as np

class HACFeatureExtractor(srk.Module):
    def __init__(self, filenames, ks, lags=[5,2], name="HACFeatureExtracter", **mfcc_params):
        super(HACFeatureExtractor, self).__init__(name, False)
        self.is_ninitilized = False
        self.filenames = filenames
        self.ks = ks
        self.lags = lags
        self.mfcc_params = mfcc_params
        
        # コードブックの作成
        cdbs = hac.build_codebooks_from_list_of_wav( self.filenames, self.ks, **self.mfcc_params )
        
        # wavをhacへ変換
        hacs = []
        for n in range(len(self.filenames)):
            hacs.append( hac.wav2hac(self.filenames[n], cdbs, self.lags, **self.mfcc_params) )
        
        # hacsを保存
        save_dir = self.get_name()
        try:
            os.mkdir( save_dir )
        except:
            pass
        np.savetxt( os.path.join( save_dir, "hac.txt"), hacs )

        self.is_ninitilized = True
    
        self.set_forward_msg( hacs )

