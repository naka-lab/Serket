#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import os

class MNVAE(srk.Module, metaclass=ABCMeta):
    def __init__( self, latent_dim, N=1, tau=1.0, rate=0.0001, epoch=5000, name="mnvae", batch_size=None, KLD_weights=[1,1], RE="BCE", load_dir=None, loss_step=50 ):
        super(MNVAE, self).__init__(name, True)
        self.__latent_dim = latent_dim
        self.__num = N
        self.__tau0 = tau
        self.__rate = rate
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__KLD_weights = KLD_weights
        self.__RE = RE
        self.__load_dir = load_dir
        self.__loss_step = loss_step
        self.__n = 0

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
        self.__pri = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        self.__tau = tf.placeholder( "float", shape=[] )
        
        # encoder
        logits_z = self.build_encoder( self.__x, self.__latent_dim )
        self.__q_z = tf.nn.softmax( logits_z )
        logits_z = tf.keras.backend.repeat( logits_z, self.__num )
                
        # zをサンプリング
        u = tf.random_uniform( tf.shape(logits_z), minval=0, maxval=1 )
        g = -tf.log( tf.clip_by_value(-tf.log(tf.clip_by_value(u, 1e-10, 1)), 1e-10, 1) )
        z = tf.nn.softmax( (logits_z + g) / self.__tau )
        self.__z = tf.reduce_sum( z, axis=1 )
        
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
        
        # lossを定義
        # KLダイバージェンス(backward message)を定義
        KLD = tf.reduce_sum( tf.multiply(self.__q_z, tf.log(tf.clip_by_value(self.__q_z, 1e-10, 1)) - tf.log(tf.clip_by_value(self.__pri, 1e-10, 1))), axis=1 )
        if self.__n==0:
            self.__loss = tf.reduce_mean( RE + self.__KLD_weights[0] * KLD )
        else:
            self.__loss = tf.reduce_mean( RE + self.__KLD_weights[1] * KLD )
            
        # lossを最小化する手法を設定
        self.__train_step = optimizer.minimize( self.__loss )

    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        # 温度係数初期値
        tau = self.__tau0
        
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
                        sess.run( self.__train_step, feed_dict={self.__x: self.__data[0], self.__pri: self.__prior, self.__tau:tau} )
                    
                    # ミニバッチ学習
                    else:
                        sff_idx = np.random.permutation(self.__N)
                        for idx in range(0, self.__N, self.__batch_size):
                            batch = self.__data[0][sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            batch_prior = self.__prior[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            sess.run( self.__train_step, feed_dict={self.__x: batch, self.__pri: batch_prior, self.__tau:tau} )
                    
                    # 指定epochごとにloss保存
                    if epoch % self.__loss_step == 0:
                        cur_loss = sess.run( self.__loss, feed_dict={self.__x: self.__data[0], self.__pri: self.__prior, self.__tau:tau} )
                        self.__loss_save.append( [epoch, cur_loss] )
                    
                    if epoch!=self.__epoch:
                        tau = np.maximum( self.__tau0*np.exp(-self.__rate*epoch), 0.5)
                    
                # サンプリング
                self.__p, self.__zz, self.__xx = sess.run( [self.__q_z, self.__z, self.__x_hat], feed_dict={self.__x: self.__data[0], self.__pri: self.__prior, self.__tau:tau} )
                # モデルの保存
                saver.save( sess, os.path.join(self.__save_dir, "model.ckpt") )
        
    def save_result( self ):
        # ｚ,x_hatを保存
        np.savetxt( os.path.join( self.__save_dir, "Pdz.txt" ), self.__p, fmt="%f" )
        np.savetxt( os.path.join( self.__save_dir, "z.txt" ), self.__zz )
        np.save( os.path.join( self.__save_dir, "x_hat.npy" ), self.__xx )
        
        # 学習時はlossを保存
        if self.__load_dir is None:
            np.savetxt( os.path.join( self.__save_dir, "loss.txt" ), self.__loss_save )

    def update( self ):
        self.__data = self.get_observations()
        self.__prior = self.get_backward_msg()

        self.__N = len(self.__data[0])                # データ数
        self.__input_dim = self.__data[0][0].shape    # 入力の次元数
        
        # backward messageがまだ計算されていないとき
        if self.__prior is None:
            self.__prior = np.ones( (self.__N, self.__latent_dim) ) / self.__latent_dim
            
        self.__data[0] = np.array( self.__data[0], dtype=np.float32 )

        if self.__load_dir is None:
            self.__save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
            self.__save_dir_ = os.path.join( self.get_name(), "%03d" % (self.__n - 1) )
        else:
            self.__save_dir = os.path.join( self.get_name(), "recog" )
        if not os.path.exists( self.__save_dir ):
            os.makedirs( self.__save_dir )

        # MNVAEの構築・学習
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
        z = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        
        # encoder
        logits = self.build_encoder( x, self.__latent_dim )
        q_z = tf.nn.softmax( logits )
        
        if self.__RE=="BCE":
            # decoder
            logits, optimizer = self.build_decoder( z )
            x_hat = tf.nn.sigmoid( logits )
        if self.__RE=="MSE":
            # decoder
            x_hat, optimizer = self.build_decoder( z )
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # モデルの読み込み
            saver.restore( sess, os.path.join(self.__load_dir, "model.ckpt") )
            # 確率計算
            q = sess.run( q_z, feed_dict={x: data[0]} )
            # zの期待値
            zz = q * self.__num
            # 再構成
            xx = sess.run( x_hat, feed_dict={z: zz} )
        
        # 結果を保存
        np.savetxt( os.path.join( save_dir, "Pdz.txt" ), q, fmt="%f" )
        np.savetxt( os.path.join( save_dir, "z.txt" ), zz )
        np.save( os.path.join( save_dir, "x_hat.npy" ), xx )

        # メッセージの送信
        self.set_forward_msg( zz )
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
        
        # zの期待値
        z = pri * self.__num
        
        if self.__RE=="BCE":
            # decoder
            logits, optimizer = self.build_decoder( z )
            x_hat = tf.nn.sigmoid( logits )
        if self.__RE=="MSE":
            # decoder
            x_hat, optimizer = self.build_decoder( z )
        
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