# encoding: utf8
#from __future__ import unicode_literals
import os
import numpy as np
import tensorflow as tf


# 変数を生成する関数を定義
def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


def save_result(z, xx, loss_save, save_dir):
    try:
        os.mkdir( save_dir )
    except:
        pass

    np.savetxt( os.path.join( save_dir, "z.txt"),z )
    np.savetxt( os.path.join( save_dir, "x_hat.txt"),xx )
    np.savetxt( os.path.join( save_dir, "loss.txt"),loss_save )


def train( data, latent_dim, weight_stddev, num_itr=5000, save_dir="model", mu_prior=None, hidden_encoder_dim=100, hidden_decoder_dim=100, batch_size=None):
    input_dim = len(data[0])
    
    a = 1.2

    # 入力を入れるplaceholder
    x = tf.placeholder("float", shape=[None, input_dim])
    mu_pri = tf.placeholder("float", shape=[None, latent_dim])
    
    # encoderの重みとバイアスを定義
    W1 = weight_variable([input_dim,hidden_encoder_dim], weight_stddev)
    b1 = bias_variable([hidden_encoder_dim])
    W2_mu = weight_variable([hidden_encoder_dim,latent_dim], weight_stddev)
    b2_mu = bias_variable([latent_dim])
    W2_logvar = weight_variable([hidden_encoder_dim,latent_dim], weight_stddev)
    b2_logvar = bias_variable([latent_dim])
    
    # 平均値を計算する関数を定義
    hidden_encoder = tf.nn.relu(tf.matmul(x, W1) + b1)
    mu_encoder = tf.matmul(hidden_encoder, W2_mu) + b2_mu
    
    # 分散を計算する関数を定義
    logvar_encoder = tf.matmul(hidden_encoder, W2_logvar) + b2_logvar
    
    # zをサンプリング
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)
    
    # decoderの重みとバイアスを定義
    W3 = weight_variable([latent_dim,hidden_decoder_dim], weight_stddev)
    b3 = bias_variable([hidden_decoder_dim])
    W4 = weight_variable([hidden_decoder_dim, input_dim], weight_stddev)
    b4 = bias_variable([input_dim])
    
    # xを生成する関数を定義
    hidden_decoder = tf.nn.relu(tf.matmul(z, W3) + b3)
    x_hat = tf.matmul(hidden_decoder, W4) + b4
    
    # ロス関数を定義
    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder - mu_pri, 2) - tf.exp(logvar_encoder), reduction_indices=1)
    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)
    loss = tf.reduce_mean(BCE + a * KLD)
    
    # lossを最小化する手法を設定
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    # 学習
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # loss保存用のlist
        loss_save = []
                
        for step in range(1, num_itr+1):
            # バッチ学習
            if batch_size==None:
                feed_dict = {x: data, mu_pri: mu_prior}
                _, cur_loss = sess.run([train_step, loss], feed_dict=feed_dict)
                # 50ごとにloss保存
                if step % 50 == 0:
                    loss_save.append([step,cur_loss])
            
            # ミニバッチ学習
            else:                
                N = len(data)
                sff_idx = np.random.permutation(N)
                for idx in range(0, N, batch_size):
                    batch = data[sff_idx[idx: idx + batch_size if idx + batch_size < N else N]]
                    batch_mu = mu_prior[sff_idx[idx: idx + batch_size if idx + batch_size < N else N]]
                    feed_dict = {x: batch, mu_pri: batch_mu}
                    _, cur_loss = sess.run([train_step, loss], feed_dict=feed_dict)
                # epochごとにloss保存
                loss_save.append([step,cur_loss])
               
        saver.save(sess, os.path.join(save_dir,"model.ckpt"))
            
    # サンプリング
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # モデル読み込み
        saver.restore(sess, os.path.join(save_dir,"model.ckpt"))
        
        zz, xx = sess.run([z, x_hat], feed_dict={x: data})
    
    # z,x_hat,loss保存
    save_result(zz, xx, loss_save, save_dir) 
    
    return zz, xx

