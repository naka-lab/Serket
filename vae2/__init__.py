#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import os

class VAE(srk.Module, metaclass=ABCMeta):
    def __init__( self, latent_dim, itr=5000, name="vae", batch_size=None, KL_param=1, RE="BCE", mode="learn" ):
        super(VAE, self).__init__(name, True)
        self.__itr = itr
        self.__latent_dim = latent_dim
        self.__batch_size = batch_size
        self.__KL_param = KL_param
        self.__RE = RE
        self.__mode = mode
        self.__save_dir = self.get_name()
        
        if RE != "BCE" and RE != "MSE":
            raise ValueError("choose RE from \"BCE\" or \"MSE\"")
            
        if mode != "learn" and mode != "recog":
            raise ValueError("choose mode from \"learn\" or \"recog\"")

    def build_model( self ):
        tf.reset_default_graph()
    
        # 入力を入れるplaceholder
        self.__x = tf.placeholder( "float", shape=[None, self.__input_dim] )
        self.__mu_pri = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        
        # encoder
        mu_encoder, logvar_encoder = self.build_encoder( self.__x, self.__input_dim, self.__latent_dim )
        
        # zをサンプリング
        epsilon = tf.random_normal( tf.shape(logvar_encoder), name='epsilon' )
        std_encoder = tf.exp( 0.5 * logvar_encoder )
        self.__z = mu_encoder + tf.multiply( std_encoder, epsilon )
        
        # decoder
        self.__x_hat, optimizer = self.build_decoder( self.__z, self.__input_dim, self.__latent_dim )
        
        # KLダイバージェンスを定義
        KLD = -0.5 * tf.reduce_sum( 1 + logvar_encoder - tf.pow(mu_encoder - self.__mu_pri, 2) - tf.exp(logvar_encoder), reduction_indices=1 )
        
        # 復元誤差を定義
        if self.__RE=="BCE":
            RE = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__x_hat, labels=self.__x), reduction_indices=1 )
        if self.__RE=="MSE":
            RE = tf.reduce_sum( tf.square(self.__x_hat - self.__x), reduction_indices=1 )
        
        # lossを定義
        self.__loss = tf.reduce_mean( RE + self.__KL_param * KLD )
        
        # lossを最小化する手法を設定
        self.__train_step = optimizer.minimize( self.__loss )

    @abstractmethod
    def build_encoder( self, x, input_dim, latent_dim ):
        pass
    
    @abstractmethod
    def build_decoder( self, z, input_dim, latent_dim ):
        pass

    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        saver = tf.train.Saver()
        
        # 学習
        if self.__mode == "learn":
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                    
                for step in range(1, self.__itr+1):
                    # バッチ学習
                    if self.__batch_size==None:
                        feed_dict = {self.__x: self.__data[0], self.__mu_pri: self.__mu_prior}
                        _, cur_loss = sess.run([self.__train_step, self.__loss], feed_dict=feed_dict)
                        # 50ごとにloss保存
                        if step % 50 == 0:
                            self.__loss_save.append([step,cur_loss])
                
                    # ミニバッチ学習
                    else:                
                        sff_idx = np.random.permutation(self.__N)
                        for idx in range(0, self.__N, self.__batch_size):
                            batch = self.__data[0][sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            batch_mu = self.__mu_prior[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            feed_dict = {self.__x: batch, self.__mu_pri: batch_mu}
                            _, cur_loss = sess.run( [self.__train_step, self.__loss], feed_dict=feed_dict )
                        # epochごとにloss保存
                        self.__loss_save.append( [step,cur_loss] )
                   
                # モデルの保存
                saver.save( sess, os.path.join(self.__save_dir, "model.ckpt") )
                
        # 認識モード時はモデルの読み込みとサンプリングのみ
        with tf.Session() as sess:
            # モデルの読み込み
            saver.restore( sess, os.path.join(self.__save_dir, "model.ckpt") )
            # サンプリング
            self.__zz, self.__xx = sess.run( [self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0]} )
        
    def save_result( self ):
        if not os.path.exists( self.__save_dir ):
            os.mkdir( self.__save_dir )
    
        # ｚ,x_hatを保存
        np.savetxt( os.path.join( self.__save_dir, "z_{}.txt".format(self.__mode) ), self.__zz )
        np.savetxt( os.path.join( self.__save_dir, "x_hat_{}.txt".format(self.__mode) ), self.__xx )
        
        # 学習モード時はlossを保存
        if self.__mode == "learn":
            np.savetxt( os.path.join( self.__save_dir, "loss.txt" ), self.__loss_save )

    def update( self ):
        self.__data = self.get_observations()
        self.__mu_prior = self.get_backward_msg()

        self.__N = len(self.__data[0])                  # データ数
        self.__input_dim = len( self.__data[0][0] )     # 入力の次元数
        
        # backward messageがまだ計算されていないとき
        if self.__mu_prior is None:
            self.__mu_prior = np.zeros( (self.__N, self.__latent_dim) )

        self.__data[0] = np.array( self.__data[0], dtype=np.float32 )

        # VAEの構築・学習
        self.build_model()
        self.train()
        
        # 結果を保存
        self.save_result()

        # メッセージの送信
        self.set_forward_msg( self.__zz )
        self.send_backward_msgs( [self.__xx] )
