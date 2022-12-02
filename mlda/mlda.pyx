# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
import random
import pickle
import os
import cython
from libc.stdlib cimport rand, RAND_MAX
import math

# ハイパーパラメータ
cdef double __alpha = 1.0
cdef double __beta = 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_lda_param( docs_mdn, topics_mdn, K, dims ):
    cdef int M = len(docs_mdn)
    cdef int D = len(docs_mdn[0])
    cdef int d, z, m, n, w, N

    # 各物体dにおいてトピックzが発生した回数
    cdef int [:,:] n_dz = np.zeros((D,K), dtype=np.int32 )

    # 各トピックzにぴおいて特徴wが発生した回数
    cdef int [:,:,:] n_mzw = np.zeros((M,K,np.max(dims)), dtype=np.int32)

    # 各トピックが発生した回数
    cdef int [:,:] n_mz = np.zeros((M,K), dtype=np.int32)

    # 数え上げる
    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            N = len(docs_mdn[m][d])    # 物体に含まれる特徴数
            for n in range(N):
                w = docs_mdn[m][d][n]       # 物体dのn番目の特徴のインデックス
                z = topics_mdn[m][d][n]     # 特徴に割り当てれれているトピック
                n_dz[d][z] += 1
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1

    return n_dz, n_mzw, n_mz
    

cdef double[:] __P = np.zeros( 100 )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sample_topic( int d, int w, int[:,:] n_dz, int[:,:] n_zw, int[:] n_z, int K, int V, double[:,:] bias_dz ):
    cdef int z

    # 累積確率を計算
    #P = (n_dz[d,:] + __alpha )*(n_zw[:,w] + __beta) / (n_z[:] + V *__beta) * bias_dz[d]

    for z in range(0,K):
        __P[z] = (n_dz[d,z] + __alpha )*(n_zw[z,w] + __beta) / (n_z[z] + V *__beta) * bias_dz[d,z]

    for z in range(1,K):
        __P[z] = __P[z] + __P[z-1]

    # サンプリング
    cdef double rnd = __P[K-1] * rand()/float(RAND_MAX)
    for z in range(K):
        if __P[z] >= rnd:
            return z

    return -1

# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数文forを回す
            doc.append(v)
    return doc

# 尤度計算
@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_liklihood( int[:,:] data, int[:,:] n_dz, int[:,:] n_zw, int[:] n_z, int D, int K, int V  ):
    cdef double lik = 0.0
    cdef double s = 0.0
    cdef double cons
    cdef int d, w, z
    
    for d in range(D):
        cons = 1/(np.sum(n_dz[d]) + K*__alpha)
        
        for w in range(V):
            s = 0.0
            
            for z in range(K):
                s += (n_zw[z][w] + __beta) / (n_z[z] + V *__beta) * (n_dz[d][z] + __alpha) * cons
            
            lik += data[d][w] * math.log(s)
    return lik

def calc_acc(results, correct):
    K = int(np.max(correct)+1)  # カテゴリ数
    N = len(results)            # データ数
    max_acc = 0                 # 精度の最大値
    changed = True              # 変化したかどうか

    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( N )

                # iとjを入れ替える
                for n in range(N):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # 精度を計算
                acc = (tmp_result==correct).sum()/float(N)

                # 精度が高くなって入れば保存
                if acc > max_acc:
                    max_acc = acc
                    results = tmp_result
                    changed = True

    return max_acc, results

# モデルの保存
def save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, categories, liks, load_dir ):
    if not os.path.exists( save_dir ):
        os.makedirs( save_dir )
    
    n_dz = np.array(n_dz)
    n_mz = np.array(n_mz)
    n_mzw = np.array(n_mzw)

    # 確率の計算と保存
    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt="%f" )
    
    # 尤度の保存
    np.savetxt( os.path.join( save_dir, "liklihood.txt" ), liks, fmt="%f" )
    
    Pmdw = []
    for m in range(M):
        Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dims[m] *__beta)
        Pdw = Pdz.dot(Pwz.T)
        Pmdw.append( Pdw )
        np.savetxt( os.path.join( save_dir, "Pmdw[{}].txt".format(m) ) , Pdw )

    if load_dir is None:
        # モデルパラメータの保存
        with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
            pickle.dump( [n_mzw, n_mz], f )
    
    # 分類結果・精度の計算と保存
    results = np.argmax( Pdz, -1 )
    if categories is not None:
        acc, results = calc_acc( results, categories )
        np.savetxt( os.path.join( save_dir, "categories.txt" ), results, fmt="%d" )
        np.savetxt( os.path.join( save_dir, "acc.txt" ), [acc], fmt="%f" )
        
    else:
        np.savetxt( os.path.join( save_dir, "categories.txt" ), results, fmt="%d" )
        
    return Pdz, Pmdw

# モデルパラメータの読み込み
def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# ldaメイン
@cython.boundscheck(False)
@cython.wraparound(False)
def train( data, K, num_itr=100, save_dir="model", bias_dz=None, categories=None, load_dir=None ):
    cdef int M, it, d, m, N, n, w, z
    cdef int[:,:] n_dz, n_mz
    cdef int[:,:,:] n_mzw
    cdef list docs_mdn, topics_mdn

    # 尤度のリスト
    liks = []
    M = len(data)       # モダリティ数

    if bias_dz is not None:
        bias_dz = np.array(bias_dz)

    dims = []
    for m in range(M):
        if data[m].all() is not None:
            dims.append( len(data[m][0]) )
            D = len(data[m])    # 物体数
        else:
            dims.append( 0 )

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_mdn = [[ None for i in range(D) ] for m in range(M)]
    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m].all() is not None:
                docs_mdn[m][d] = conv_to_word_list( data[m][d] )
                topics_mdn[m][d] = np.random.randint( 0, K, len(docs_mdn[m][d]) ) # 各単語にランダムでトピックを割り当てる

    # LDAのパラメータを計算
    n_dz, n_mzw, n_mz = calc_lda_param( docs_mdn, topics_mdn, K, dims )

    # 認識モードの時は学習したパラメータを読み込み
    if load_dir is not None:
        n_mzw, n_mz = load_model( load_dir )

    for it in range(num_itr):
        # メインの処理
        for d in range(D):
            for m in range(M):
                if data[m].all() is None:
                    continue

                N = len(docs_mdn[m][d]) # 物体dのモダリティmに含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]       # 特徴のインデックス
                    z = topics_mdn[m][d][n]     # 特徴に割り当てられているカテゴリ


                    # データを取り除きパラメータを更新
                    n_dz[d][z] -= 1

                    if load_dir is None:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1

                    # サンプリング
                    z = sample_topic( d, w, n_dz, n_mzw[m], n_mz[m], K, dims[m], bias_dz )

                    # データをサンプリングされたクラスに追加してパラメータを更新
                    topics_mdn[m][d][n] = z
                    n_dz[d][z] += 1

                    if load_dir is None:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1

        # 尤度計算
        lik = 0
        for m in range(M):
            if data[m].all() is not None:
                lik += calc_liklihood( data[m], n_dz, n_mzw[m], n_mz[m], D, K, dims[m] )
        liks.append( lik )
        
    params = save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, categories, liks, load_dir )
    
    return params