#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import os

class TorchVAE(srk.Module, metaclass=ABCMeta):
    def __init__( self, input_dim, latent_dim, lr=1e-3, device="cpu", epoch=5000, name="vae", batch_size=None, KLD_weight=1, RE="BCE", load_dir=None, loss_step=50 ):
        super(TorchVAE, self).__init__(name, True)
        self.__epoch = epoch
        self.__latent_dim = latent_dim
        self.__batch_size = batch_size
        self.__KLD_weight = KLD_weight
        self.__RE = RE
        self.__load_dir = load_dir
        self.__loss_step = loss_step
        self.__n = 0
        self.__lr = lr
        self.device = device
        
        if RE != "BCE" and RE != "MSE":
            raise ValueError("choose RE from \"BCE\" or \"MSE\"")

    def tensor( self, x ):
        return torch.tensor( x, dtype=torch.float32, device=self.device )
    
    def numpy( self, x ):
        return x.cpu().detach().numpy()

    @abstractmethod
    def build_encoder( self, input_dim, latent_dim ):
        pass
    
    @abstractmethod
    def build_decoder( self, input_dim, latent_dim ):
        pass

    def build_model( self ):
        self.__encoder = self.build_encoder( self.__input_dim, self.__latent_dim ).to(self.device)
        self.__decoder = self.build_decoder( self.__input_dim, self.__latent_dim ).to(self.device)

        if self.__RE=="BCE":
            self.__loss_func = nn.BCELoss(reduction="sum")
        if self.__RE=="MSE":
            self.__loss_func = nn.MSELoss(reduction="sum")

    def foward( self, x ):
        mu, log_var = self.__encoder( x )
        eps = torch.randn_like(log_var)
        z = mu + torch.exp(log_var/2) * eps
        x_decode = self.__decoder( z )

        return mu, log_var, z, x_decode

    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        # 学習
        if self.__load_dir is None:
                optimizer = torch.optim.Adam( list(self.__encoder.parameters())+list(self.__decoder.parameters()), lr=self.__lr)
               
                for epoch in range(1, self.__epoch+1):
                    if self.__batch_size is None:
                        batch_size = self.__N
                    else:
                        batch_size = self.__batch_size

                    sff_idx = np.random.permutation(self.__N)

                    total_loss = 0
                    for idx in range(0, self.__N, batch_size):
                        batch = self.__data[0][sff_idx[idx: idx + batch_size if idx + batch_size < self.__N else self.__N]]
                        batch_mu = self.__mu_prior[sff_idx[idx: idx + batch_size if idx + batch_size < self.__N else self.__N]]

                        mu, log_var, z, x_decode = self.foward( batch )

                        reconst_error = self.__loss_func(x_decode, batch )
                        kl = - 0.5 * torch.sum(1 + log_var - (mu-batch_mu)**2 - torch.exp(log_var) )
                        loss = reconst_error + self.__KLD_weight * kl

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += self.numpy(loss)
                    self.__loss_save.append( total_loss )

                # サンプリング
                #self.__zz, self.__xx = sess.run( [self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0]} )
                self.__zz, _, _, self.__xx = self.foward( self.__data[0] )

                # モデルの保存
                #saver.save( sess, os.path.join(self.__save_dir, "model.ckpt") )
                torch.save( self.__encoder.state_dict(), os.path.join(self.__save_dir, "encoder.pth") )
                torch.save( self.__decoder.state_dict(), os.path.join(self.__save_dir, "decoder.pth") )
        # 認識
        else:
            # モデルの読み込み
            #saver.restore( sess, os.path.join(self.__load_dir, "model.ckpt") )
            self.__encoder.load_state_dict( torch.load( os.path.join(self.__load_dir, "encoder.pth") ) )
            self.__decoder.load_state_dict( torch.load( os.path.join(self.__load_dir, "decoder.pth") ) )

            # サンプリング
            #self.__zz, self.__xx = sess.run( [self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0]} )
            self.__zz, _, _, self.__xx =self.foward( self.__data[0] )

       
    def save_result( self ):
        # ｚ,x_hatを保存
        np.savetxt( os.path.join( self.__save_dir, "z.txt" ), self.numpy(self.__zz) )
        np.save( os.path.join( self.__save_dir, "x_hat.npy" ), self.numpy(self.__xx) )
        
        # 学習時はlossを保存
        if self.__load_dir is None:
            np.savetxt( os.path.join( self.__save_dir, "loss.txt" ), self.__loss_save )

    def update( self ):
        self.__data =  self.get_observations()
        self.__mu_prior =  self.get_backward_msg()

        self.__N = len(self.__data[0])                # データ数
        self.__input_dim = self.__data[0][0].shape    # 入力の次元数
        
        # backward messageがまだ計算されていないとき
        if self.__mu_prior is None:
            self.__mu_prior = torch.zeros( (self.__N, self.__latent_dim), device=self.device )

        self.__mu_prior = self.tensor(self.__mu_prior)

        self.__data[0] = self.tensor(self.__data[0])

        if self.__load_dir is None:
            self.__save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
            self.__save_dir_ = os.path.join( self.get_name(), "%03d" % (self.__n - 1) )
        else:
            self.__save_dir = os.path.join( self.get_name(), "recog" )
        if not os.path.exists( self.__save_dir ):
            os.makedirs( self.__save_dir )

        # VAEの構築・学習
        self.build_model()
        self.train()
        
        # 結果を保存
        self.save_result()
        
        self.__n += 1

        # メッセージの送信
        self.set_forward_msg( self.numpy(self.__zz) )
        self.send_backward_msgs( [self.numpy(self.__xx)] )

    def recog( self ):
        data = self.get_observations()

        input_dim = data[0][0].shape    # 入力の次元数
            
        save_dir = os.path.join( self.get_name(), "recog" )
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )

        """
        # グラフのリセット
        tf.reset_default_graph()
        
        # 入力を入れるplaceholder
        x = tf.placeholder( "float", shape=np.concatenate([[None], input_dim]) )
        
        # encoder
        mu, _ = self.build_encoder( x, self.__latent_dim )
        
        if self.__RE=="BCE":
            # decoder
            logits, optimizer = self.build_decoder( mu )
            x_hat = tf.nn.sigmoid( logits )
        if self.__RE=="MSE":
            # decoder
            x_hat, optimizer = self.build_decoder( mu )
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # モデルの読み込み
            saver.restore( sess, os.path.join(self.__load_dir, "model.ckpt") )
            # 再構成
            xx = sess.run( x_hat, feed_dict={x: data[0]} )
        """


        self.__encoder.load_state_dict( torch.load( os.path.join(self.__load_dir, "encoder.pth") ) )
        self.__decoder.load_state_dict( torch.load( os.path.join(self.__load_dir, "decoder.pth") ) )

        mu, _ = self.__encoder( self.tensor( self.__data[0] ) )
        xx = self.__decoder( mu )
        
        # 結果を保存
        np.savetxt( os.path.join( save_dir, "z.txt" ), mu )
        np.save( os.path.join( save_dir, "x_hat.npy" ), xx )

        # メッセージの送信
        self.set_forward_msg( mu )
        self.send_backward_msgs( [xx] )

    def predict( self ):
        data = self.get_observations()
        prior = self.get_backward_msg()

        input_dim = data[0][0].shape    # 入力の次元数
            
        save_dir = os.path.join( self.get_name(), "predict" )
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )

        """
        # グラフのリセット
        tf.reset_default_graph()
        
        # 入力を入れるplaceholder
        x = tf.placeholder( "float", shape=np.concatenate([[None], input_dim]) )
        pri = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        
        # encoder
        self.build_encoder( x, self.__latent_dim )
        
        if self.__RE=="BCE":
            # decoder
            logits, optimizer = self.build_decoder( pri )
            x_hat = tf.nn.sigmoid( logits )
        if self.__RE=="MSE":
            # decoder
            x_hat, optimizer = self.build_decoder( pri )
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # モデルの読み込み
            saver.restore( sess, os.path.join(self.__load_dir, "model.ckpt") )
            # サンプリング
            xx = sess.run( x_hat, feed_dict={pri: prior} )
        """

        prior = self.tensor( prior )
        self.__decoder.load_state_dict( torch.load( os.path.join(self.__load_dir, "decoder.pth") ) )
        xx = self.__decoder( prior )
        
        # 結果を保存
        np.save( os.path.join( save_dir, "x_hat.npy" ), self.numpy(xx) )

        # メッセージの送信
        self.send_backward_msgs( [xx] )