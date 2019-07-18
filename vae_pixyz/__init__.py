#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import torch
from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler, StochasticReconstructionLoss
from pixyz.models import Model
from abc import ABCMeta, abstractmethod
import os

class VAE(srk.Module, metaclass=ABCMeta):
    def __init__( self, latent_dim, itr=5000, name="vae", batch_size=None, KL_param=1, mode="learn" ):
        super(VAE, self).__init__(name, True)
        self.__latent_dim = latent_dim
        self.__itr = itr
        self.__batch_size = batch_size
        self.__KL_param = KL_param
        self.__mode = mode
            
        if mode != "learn" and mode != "recog":
            raise ValueError("choose mode from \"learn\" or \"recog\"")
        
        if torch.cuda.is_available():
            self.__device = "cuda"
        else:
            self.__device = "cpu"

    @abstractmethod
    def network( self, input_dim, latent_dim ):
        class Inference(Normal):
            def __init__( self ):
                pass
            
            def forward( self, x ):
                pass
        
        class Generator(Normal):
            def __init__( self ):
                pass
            
            def forward( self, z ):
                pass

    def build_model(self):
        class Prior(Normal):
            def __init__( self ):
                super(Prior, self).__init__(cond_var=["mu", "sigma"], var=["z"], name="p_{prior}")
                
            def forward(self, mu, sigma):
                return {"loc": mu, "scale": sigma}
        
        # priorの生成
        prior = Prior()
        
        # network構造の取得
        self.__q, self.__p, optimizer, op_param = self.network( self.__input_dim, self.__latent_dim )
        
        # lossの定義
        loss = ( self.__KL_param*KullbackLeibler(self.__q, prior) + StochasticReconstructionLoss(self.__q, self.__p) ).mean()
        
        # modelの構築
        self.__model = Model( loss=loss, distributions=[self.__p, self.__q], optimizer=optimizer, optimizer_params=op_param )
        
    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        # 学習
        if self.__mode == "learn":
            for step in range(1, self.__itr+1):
                # バッチ学習
                if self.__batch_size==None:
                    cur_loss = self.__model.train( {"x": self.__data[0], "mu": self.__mu, "sigma": self.__sigma} )
                    # 50ごとにloss保存
                    if step % 50 == 0:
                        self.__loss_save.append( [step, cur_loss] )
            
                # ミニバッチ学習
                else:                
                    sff_idx = np.random.permutation(self.__N)
                    for idx in range(0, self.__N, self.__batch_size):
                        batch = self.__data[0][sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                        batch_mu = self.__mu[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                        batch_sigma = self.__sigma[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                        cur_loss = self.__model.train( {"x": batch, "mu": batch_mu, "sigma": batch_sigma} )
                    # epochごとにloss保存
                    self.__loss_save.append( [step, cur_loss] )
                    
            # encoder保存
            torch.save( self.__q.state_dict(), os.path.join(self.__save_dir, "encoder") )
            # decoder保存
            torch.save( self.__p.state_dict(), os.path.join(self.__save_dir, "decoder") )
               
        # 認識モード時はモデルの読み込み
        if self.__mode == "recog":
            # encoderの読み込み
            self.__q.load_state_dict( torch.load( os.path.join(self.__save_dir, "encoder") ) )
            # decoderの読み込み
            self.__p.load_state_dict( torch.load( os.path.join(self.__save_dir, "decoder") ) )
        
        # サンプリング
        self.__zz = self.__q.sample( self.__data[0], return_all=False )
        self.__xx = self.__p.sample_mean( self.__zz )
        # ndarrayに変換
        self.__zz = self.__zz["z"].cpu().numpy()
        self.__xx = self.__xx.cpu().detach().numpy()

    def save_result( self ):
        # ｚ,x_hatを保存
        np.savetxt( os.path.join( self.__save_dir, "z_{}.txt".format(self.__mode) ), self.__zz )
        np.savetxt( os.path.join( self.__save_dir, "x_hat_{}.txt".format(self.__mode) ), self.__xx )
        
        # 学習モード時はlossを保存
        if self.__mode == "learn":
            np.savetxt( os.path.join( self.__save_dir, "loss.txt" ), self.__loss_save )

    def update( self, i ):
        self.__data = self.get_observations()
        back_msg = self.get_backward_msg()              # [Nomals, calasses]

        self.__N = len(self.__data[0])                  # データ数
        
        if len(self.__data[0].shape)==2:
            self.__input_dim = self.__data[0][0].shape  # 入力の次元数
        else:
            self.__input_dim = self.__data[0][0].shape  # 時系列データ(系列長*次元数)or画像の場合(高さ*幅*チャンネル)
        
        # backward messageがまだ計算されていないとき
        if back_msg is None:
            self.__mu = np.zeros( (self.__N, self.__latent_dim) )
            self.__sigma = np.ones( (self.__N, self.__latent_dim) )
        # データ点のクラスに対応したmuとsigmaを格納[N, latent_dim]
        else:
            mus = [ back_msg[0][i].loc[0] for i in range(len(back_msg[0])) ]
            sigmas = [ back_msg[0][i].scale[0] for i in range(len(back_msg[0])) ]
            self.__mu = [ mus[c].cpu().detach().numpy() for c in back_msg[1] ]
            self.__sigma = [ sigmas[c].cpu().detach().numpy() for c in back_msg[1] ]
        
        self.__data[0] = torch.Tensor( self.__data[0] ).to(self.__device)
        self.__mu = torch.Tensor( self.__mu ).to(self.__device)
        self.__sigma = torch.Tensor( self.__sigma ).to(self.__device)

        self.__save_dir = self.get_name()+"_%d"%i
        
        # 保存フォルダの作成
        if not os.path.exists( self.__save_dir ):
            os.mkdir( self.__save_dir )
        
        # VAEの構築・学習
        self.build_model()
        self.train()

        # 結果を保存
        self.save_result()

        # メッセージの送信
        self.set_forward_msg( self.__zz )
        self.send_backward_msgs( [self.__xx] )
