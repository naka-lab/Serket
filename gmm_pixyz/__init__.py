#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import torch
from pixyz.distributions import Normal, Categorical
from pixyz.distributions.mixture_distributions import MixtureModel 
import os
from copy import deepcopy

class GMM(srk.Module):
    def __init__( self, K, itr=100, name="gmm", category=None, mode="learn" ):
        super(GMM, self).__init__(name, True)
        self.__K = K
        self.__itr = itr
        self.__category = category
        self.__mode = mode
            
        if mode != "learn" and mode != "recog":
            raise ValueError("choose mode from \"learn\" or \"recog\"")
        
        if torch.cuda.is_available():
            self.__device = "cuda"
        else:
            self.__device = "cpu"

    def build_model( self ):
        # P(x|z)の作成
        self.__distributions = []
        for i in range(self.__K):
            loc = torch.randn( self.__input_dim ).to(self.__device)
            scale = torch.empty( self.__input_dim ).fill_(0.5).to(self.__device)
            self.__distributions.append( Normal(loc=loc, scale=scale, var=["x"], name="p_%d" %i).to(self.__device) )
        
        # P(z)の作成
        self.__prior = Categorical( probs=self.__prior_probs, var=["z"], name="p_{prior}" ).to(self.__device)
        
        # P(x)の作成
        self.__p = MixtureModel( distributions=self.__distributions, prior=self.__prior ).to(self.__device)
        
        # P(z|x)の作成        
        self.__post = self.__p.posterior()
        
    def train( self ):
        # E-step
        self.__posterior = self.__post.prob().eval({"x": self.__data[0]})
        
        # M-step
        N_k = self.__posterior.sum(dim=1)
        
        # update probs
        probs = N_k / N_k.sum()
        self.__prior.probs[0] = probs

        # update loc & scale
        loc = (self.__posterior[:, None] @ self.__data[0][None]).squeeze(1)
        loc /= (N_k[:, None] + 1e-6)
    
        cov = (self.__data[0][None, :, :] - loc[:, None, :]) ** 2
        var = (self.__posterior[:, None, :] @ cov).squeeze(1)
        var /= (N_k[:, None] + 1e-6)
        scale = var.sqrt()
    
        for i, d in enumerate(self.__distributions):
            d.loc[0] = loc[i]
            d.scale[0] = scale[i]

    def sampling( self ):
        # 事後分布からサンプリングできるように転置・カテゴリカル分布の生成
        self.__post_probs = torch.t( self.__posterior )
        class_dist = Categorical( probs=self.__post_probs )
        
        # classをone-hotからintに変換
        one_hot_class = class_dist.sample()["x"].cpu().detach().numpy()[0]
        self.__classes = [np.where(r==1)[0][0] for r in one_hot_class]
        
    def calc_acc( self ):
        self.__max_acc = 0  # 精度の最大値
        changed = True      # 変化したかどうか
        self.__rp_classes = deepcopy( self.__classes ) # replace用にclassesをコピー
    
        while changed:
            changed = False
            for i in range(self.__K):
                for j in range(self.__K):
                    tmp_result = np.zeros( self.__N )
    
                    # iとjを入れ替える
                    for n in range(self.__N):
                        if self.__rp_classes[n]==i: tmp_result[n]=j
                        elif self.__rp_classes[n]==j: tmp_result[n]=i
                        else: tmp_result[n] = self.__rp_classes[n]
    
                    # 精度を計算
                    acc = (tmp_result==self.__category).sum() / float(self.__N)
    
                    # 精度が高くなって入れば保存
                    if acc > self.__max_acc:
                        self.__max_acc = acc
                        self.__rp_classes = tmp_result
                        changed = True
    
    def save_model( self ):
        if not os.path.exists( self.__save_dir ):
            os.mkdir( self.__save_dir )
    
        # モデルパラメータの保存
        if self.__mode == "learn":
            # 確率と平均の保存
            loc = []
            scale = []
            for k in range(self.__K):
                loc.append( self.__distributions[k].loc[0].cpu().detach().numpy() )
                scale.append( self.__distributions[k].scale[0].cpu().detach().numpy() )
            np.savetxt( os.path.join( self.__save_dir, "mu.txt" ), loc, fmt=str("%f") )
            np.savetxt( os.path.join( self.__save_dir, "sigma.txt" ), scale, fmt=str("%f") )
    
        # 分類結果・精度の計算・保存
        if self.__category is not None:
            self.calc_acc()
            np.savetxt( os.path.join( self.__save_dir, "class_{}.txt".format(self.__mode) ), self.__rp_classes, fmt=str("%d") )
            np.savetxt( os.path.join( self.__save_dir, "acc_{}.txt".format(self.__mode) ), [self.__max_acc], fmt=str("%f") )
            
        else:
            np.savetxt( os.path.join( self.__save_dir, "class{}.txt".format(self.__mode) ), self.__classes, fmt=str("%d") )

    def load_model( self ):
        loc = np.loadtxt("mu.txt")
        scale = np.loadtxt("sigma.txt")
        for i, d in enumerate(self.__distributions):
            d.loc[0] = loc[i].to(self.__device)
            d.scale[0] = scale[i].to(self.__device)
    
    def recog( self ):
        self.load_model()
        self.__posterior = self.__post.prob().eval({"x": self.__data[0]})

    def update( self, i ):
        self.__data = self.get_observations()
        self.__prior_probs = self.get_backward_msg()    # バックワードされた事前分布

        self.__N = len(self.__data[0])                  # データ数
        self.__input_dim = len( self.__data[0][0] )     # 入力の次元数
        
        # backward messageがまだ計算されていないとき
        if self.__prior_probs is None:
            self.__prior_probs = torch.empty( self.__K ).fill_( 1. / self.__K ).to(self.__device) # 一様分布

        self.__data[0] = torch.Tensor( self.__data[0] ).to(self.__device)
        
        self.__save_dir = self.get_name()+"_%d"%i
        
        # VAEの構築
        self.build_model()
        
        # 学習or認識
        if self.__mode=="learn":
            for i in range(self.__itr):
                self.train()
        else:
            self.recog()
        
        # サンプリング
        self.sampling()

        # 結果を保存
        self.save_model()

        # メッセージの送信
        self.set_forward_msg( self.__post )
        self.send_backward_msgs( [[self.__distributions, self.__classes]] )
