#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import os

class MVAE(srk.Module, metaclass=ABCMeta):
    def __init__( self, latent_dim, itr=5000, name="vae", batch_size=None, KL_param=[1,1], RE="BCE", load_dir=None ):
        super(MVAE, self).__init__(name, True)
        self.__itr = itr
        self.__latent_dim = latent_dim
        self.__batch_size = batch_size
        self.__KL_param = KL_param
        self.__RE = RE
        self.__load_dir = load_dir
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
        
        # 事前分布のパラメータ計算
        self.__a = np.ones((1 , self.__latent_dim)).astype(np.float32)
        self.__mu_prior = tf.constant( ( np.log(self.__a).T - np.mean(np.log(self.__a),1) ).T )
        self.__var_prior = tf.constant( ( ((1.0/self.__a)*(1 - (2.0/self.__latent_dim))).T + (1.0/(self.__latent_dim**2))*np.sum(1.0/self.__a,1) ).T )
        
        # 入力を入れるplaceholder
        self.__x = tf.placeholder( "float", shape=np.concatenate([[None], self.__input_dim]) )
        self.__pri = tf.placeholder( "float", shape=[None, self.__latent_dim] )
        
        # encoder
        mu, logvar = self.build_encoder( self.__x, self.__latent_dim )
        
        # zをサンプリング
        epsilon = tf.random_normal( tf.shape(logvar), name='epsilon' )
        std = tf.exp( 0.5 * logvar )
        self.__z_pri = mu + tf.multiply( std, epsilon )
        self.__z = tf.nn.softmax( self.__z_pri )
        
        # decoder
        if self.__RE=="BCE":
            logits, optimizer = self.build_decoder( self.__z )
            self.__x_hat = tf.nn.sigmoid( logits )
        if self.__RE=="MSE":
            self.__x_hat, optimizer = self.build_decoder( self.__z )
        
        # KLダイバージェンスを定義
        # ラプラス近似の項
        KLD1 = 0.5 * ( tf.reduce_sum( tf.div(tf.exp(logvar), self.__var_prior) 
                                      + tf.multiply( tf.div((self.__mu_prior - mu), self.__var_prior), (self.__mu_prior - mu) ) 
                                      + tf.log(self.__var_prior) - logvar, axis=1 )
                      - self.__latent_dim )
        # backward messageの項
        KLD2 = tf.reduce_sum( tf.multiply(self.__z, tf.log(self.__z) - tf.log(self.__pri)), axis=1 )
        
        # 復元誤差を定義
        F = tf.keras.layers.Flatten()
        if self.__RE=="BCE":
            RE = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=F(logits), labels=F(self.__x)), axis=1 )
        if self.__RE=="MSE":
            RE = tf.keras.losses.mean_squared_error( F(self.__x), F(self.__x_hat) )
        
        # lossを定義
        if self.__n==0:
            self.__loss = tf.reduce_mean( RE + self.__KL_param[0] * KLD1 )
        else:
            self.__loss = tf.reduce_mean( RE + self.__KL_param[0] * KLD1 + self.__KL_param[1] * KLD2 )
            
        # lossを最小化する手法を設定
        self.__train_step = optimizer.minimize( self.__loss )

    def train( self ):
        # loss保存用のlist
        self.__loss_save = []
        
        saver = tf.train.Saver()
        
        # 学習
        if self.__load_dir is None:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                    
                for step in range(1, self.__itr+1):
                    # バッチ学習
                    if self.__batch_size is None:
                        feed_dict = {self.__x: self.__data[0], self.__pri: self.__prior}
                        _, cur_loss = sess.run([self.__train_step, self.__loss], feed_dict=feed_dict)
                        # 50ごとにloss保存
                        if step % 50 == 0:
                            self.__loss_save.append([step,cur_loss])
                
                    # ミニバッチ学習
                    else:                
                        sff_idx = np.random.permutation(self.__N)
                        for idx in range(0, self.__N, self.__batch_size):
                            batch = self.__data[0][sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            batch_prior = self.__prior[sff_idx[idx: idx + self.__batch_size if idx + self.__batch_size < self.__N else self.__N]]
                            feed_dict = {self.__x: batch, self.__pri: batch_prior}
                            _, cur_loss, z = sess.run( [self.__train_step, self.__loss, self.__z_pri], feed_dict=feed_dict )
                        # epochごとにloss保存
                        self.__loss_save.append( [step,cur_loss] )
                   
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
        self.__prior = self.get_backward_msg()

        self.__N = len(self.__data[0])                      # データ数
        
        if len(self.__data[0].shape)==2:
            self.__input_dim = self.__data[0][0].shape      # 入力の次元数
        else:
            self.__input_dim = self.__data[0][0].shape      # 時系列データ(系列長*次元数)or画像の場合(高さ*幅*チャンネル)
        
        # backward messageがまだ計算されていないとき
        if self.__prior is None:
            self.__prior = np.ones( (self.__N, self.__latent_dim) ) / self.__latent_dim
            
        self.__data[0] = np.array( self.__data[0], dtype=np.float32 )

        if self.__load_dir is None:
            self.__save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
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
