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
    def __init__( self, latent_dim, epoch=5000, name="vae", batch_size=None, KLD_weight=1, RE="BCE", load_dir=None, loss_step=50 ):
        super(VAE, self).__init__(name, True)
        self.__epoch = epoch
        self.__latent_dim = latent_dim
        self.__batch_size = batch_size
        self.__KLD_weight = KLD_weight
        self.__RE = RE
        self.__load_dir = load_dir
        self.__loss_step = loss_step
        self.__n = 0
        
        if RE != "BCE" and RE != "MSE":
            raise ValueError("choose RE from \"BCE\" or \"MSE\"")
            
    @abstractmethod
    def build_encoder( self, x, latent_dim ):
        pass
    
    @abstractmethod
    def build_decoder( self, z ):
        pass

    def build_model( self ):
        tf.reset_default_graph()
    
        # 入力を入れるplaceholder
        self.__x = tf.placeholder( "float", shape=np.concatenate([[None], self.__input_dim]) )
        self.__mu_pri = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        
        # encoder
        mu, logvar = self.build_encoder( self.__x, self.__latent_dim )
        
        # zをサンプリング
        epsilon = tf.random_normal( tf.shape(logvar), name='epsilon' )
        std = tf.exp( 0.5 * logvar )
        self.__z = mu + tf.multiply( std, epsilon )
        
        F = tf.keras.layers.Flatten()
        if self.__RE=="BCE":
            # decoder
            logits, optimizer = self.build_decoder( self.__z )
            self.__x_hat = tf.nn.sigmoid( logits )
            # reconstruction error
            RE = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=F(logits), labels=F(self.__x)), axis=1 )
        if self.__RE=="MSE":
            # decoder
            self.__x_hat, optimizer = self.build_decoder( self.__z )
            # reconstruction error
            RE = tf.reduce_sum( tf.math.squared_difference(F(self.__x), F(self.__x_hat)), axis=1 )
        
        # KLダイバージェンスを定義
        KLD = -0.5 * tf.reduce_sum( 1 + logvar - tf.pow(mu - self.__mu_pri, 2) - tf.exp(logvar), axis=1 )
        
        # lossを定義
        self.__loss = tf.reduce_mean( RE + self.__KLD_weight * KLD )
        
        # lossを最小化する手法を設定
        self.__train_step = optimizer.minimize( self.__loss )

    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        saver = tf.train.Saver()
        
        # 学習
        if self.__load_dir is None:
            with tf.Session() as sess:
#                sess.run(tf.global_variables_initializer())
                if self.__n==0:
                    sess.run(tf.global_variables_initializer())
                else:
                    saver.restore( sess, os.path.join(self.__save_dir_, "model.ckpt") )
                
                for epoch in range(1, self.__epoch+1):
                    # バッチ学習
                    if self.__batch_size is None:
                        sess.run( self.__train_step, feed_dict={self.__x: self.__data[0], self.__mu_pri: self.__mu_prior} )
                
                    # ミニバッチ学習
                    else:                
                        sff_idx = np.random.permutation(self.__N)
                        for idx in range(0, self.__N, self.__batch_size):
                            batch = self.__data[0][sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            batch_mu = self.__mu_prior[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            sess.run( self.__train_step, feed_dict={self.__x: batch, self.__mu_pri: batch_mu} )
                    
                    # 指定epochごとにloss保存
                    if epoch % self.__loss_step == 0:
                        cur_loss = sess.run( self.__loss, feed_dict={self.__x: self.__data[0], self.__mu_pri: self.__mu_prior} )
                        self.__loss_save.append( [epoch, cur_loss] )

                # サンプリング
                self.__zz, self.__xx = sess.run( [self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0]} )
                # モデルの保存
                saver.save( sess, os.path.join(self.__save_dir, "model.ckpt") )
        # 認識
        else:
            with tf.Session() as sess:
                # モデルの読み込み
                saver.restore( sess, os.path.join(self.__load_dir, "model.ckpt") )
                # サンプリング
                self.__zz, self.__xx = sess.run( [self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0]} )
        
    def save_result( self ):
        # ｚ,x_hatを保存
        np.savetxt( os.path.join( self.__save_dir, "z.txt" ), self.__zz )
        np.save( os.path.join( self.__save_dir, "x_hat.npy" ), self.__xx )
        
        # 学習時はlossを保存
        if self.__load_dir is None:
            np.savetxt( os.path.join( self.__save_dir, "loss.txt" ), self.__loss_save )

    def update( self ):
        self.__data = self.get_observations()
        self.__mu_prior = self.get_backward_msg()

        self.__N = len(self.__data[0])                # データ数
        self.__input_dim = self.__data[0][0].shape    # 入力の次元数
        
        # backward messageがまだ計算されていないとき
        if self.__mu_prior is None:
            self.__mu_prior = np.zeros( (self.__N, self.__latent_dim) )

        self.__data[0] = np.array( self.__data[0], dtype=np.float32 )

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
        self.set_forward_msg( self.__zz )
        self.send_backward_msgs( [self.__xx] )

    def recog( self ):
        data = self.get_observations()

        input_dim = data[0][0].shape    # 入力の次元数
            
        save_dir = os.path.join( self.get_name(), "recog" )
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )

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
        
        # 結果を保存
        np.save( os.path.join( save_dir, "x_hat.npy" ), xx )

        # メッセージの送信
        self.send_backward_msgs( [xx] )