# encoding: utf8
#from __future__ import unicode_literals
import os
import numpy as np
import tensorflow as tf

def save_result( message, loss_save, loss_save_, save_dir, load_dir ):
    # messageを保存
    np.savetxt( os.path.join( save_dir, "message.txt" ), message, fmt="%f" )
    
    # lossを保存，学習時は元のモデルのlossも保存
    if load_dir is None:
        np.savetxt( os.path.join( save_dir, "loss.txt" ), loss_save )
    np.savetxt( os.path.join( save_dir, "loss_message.txt" ), loss_save_ )

def train( data, x, y, m_, graph, loss, train_step, index, num_itr1=5000, num_itr2=5000, save_dir="model", batch_size1=None, batch_size2=None, seaquence_size=None, load_dir=None ):
    N = len(data[0])        # データ数
    D_x = len(data[0][0])   # 入力の次元数
    D_y = len(data[1][0])   # 出力の次元数
    
    # loss保存用のlist
    loss_save = []   # 元のモデル用
    loss_save_ = []  # messageモデル用
    
    # 元のモデルの学習
    if load_dir is None:
        with tf.Session(graph=graph[0]) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
                
            for step in range(1, num_itr1+1):
                # バッチ学習
                if batch_size1==None:
                    _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: data[0], y[0]: data[1]})
                    # 50ごとにloss保存
                    if step % 50 == 0:
                        loss_save.append([step,cur_loss])
                        
                # ミニバッチ学習
                else:
                    if seaquence_size==None:
                        # ランダムインデックスを作成
                        sff_idx = np.random.permutation(N)
                        for idx in range(0, N, batch_size1):
                            # ランダムインデックスをバッチサイズに小分け
                            idx_ = sff_idx[idx: idx + batch_size1 if idx + batch_size1 < N else N]
                            # ミニバッチ作成
                            batch_x = data[0][idx_]
                            batch_y = data[1][idx_]
                            _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: batch_x, y[0]: batch_y})
                    else:
                        # シーケンスサイズのステップでランダムインデックスを作成
                        sff_idx_ = np.arange(0, N, seaquence_size)
                        np.random.shuffle(sff_idx_)
                        sff_idx = np.array([np.arange(sff_idx_[i], sff_idx_[i]+seaquence_size) for i in range(len(sff_idx_))]).flatten()
                        for idx in range(0, N, batch_size1):
                            # ランダムインデックスをバッチサイズに小分け
                            idx_ = sff_idx[idx: idx + batch_size1 if idx + batch_size1 < N else N]
                            # ミニバッチ作成
                            batch_x = data[0][idx_]
                            batch_y = data[1][idx_]
                            _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: batch_x, y[0]: batch_y})
                        
                    # epochごとにloss保存
                    loss_save.append([step,cur_loss])
                
            # モデルの保存
            saver.save(sess, os.path.join(save_dir, "model.ckpt"))

    # message学習
    with tf.Session(graph=graph[1]) as sess:
        sess.run(tf.global_variables_initializer())
        # m(TRAINABLE_VARIABLES[0])以外の変数を復元するため引数を指定
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1:])
        
        # モデルの読み込み(m以外のパラメータを復元)
        if load_dir is None:
            saver.restore(sess, os.path.join(save_dir, "model.ckpt"))
        else:
            saver.restore(sess, os.path.join(load_dir, "model.ckpt"))
        
        for step in range(1, num_itr2+1):
            # バッチ学習
            if batch_size2==None:
                _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: data[0], y[1]: data[1]})
                # 50ごとにloss保存
                if step % 50 == 0:
                    loss_save_.append([step,cur_loss_])
            
            # ミニバッチ学習
            else:
                if seaquence_size==None:
                    # ランダムインデックスを作成
                    sff_idx = np.random.permutation(N)
                    for idx in range(0, N, batch_size2):
                        # ランダムインデックスをバッチサイズに小分け
                        idx_ = sff_idx[idx: idx + batch_size2 if idx + batch_size2 < N else N]
                        # ミニバッチ作成
                        batch_x = data[0][idx_]
                        batch_y = data[1][idx_]
                        _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: batch_x, y[1]: batch_y, index: idx_})
                else:
                    # シーケンスサイズのステップでランダムインデックスを作成
                    sff_idx = np.arange(0, N, seaquence_size)
                    np.random.shuffle(sff_idx)
                    sff_idx = np.array([np.arange(sff_idx_[i], sff_idx_[i]+seaquence_size) for i in range(len(sff_idx_))]).flatten()
                    for idx in range(0, N, batch_size2):
                        # ランダムインデックスをバッチサイズに小分け
                        idx_ = sff_idx[idx: idx + batch_size2 if idx + batch_size2 < N else N]
                        # ミニバッチ作成
                        batch_x = data[0][idx_]
                        batch_y = data[1][idx_]
                        _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: batch_x, y[1]: batch_y, index: idx_})
                        
                # epochごとにloss保存
                loss_save_.append([step,cur_loss_])

        # messageを出力
        message = sess.run([m_])[0]
        
    # 結果を保存
    save_result( message, loss_save, loss_save_, save_dir, load_dir ) 
    
    return message
