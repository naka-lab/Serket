# encoding: utf-8
import numpy
import random
import math
import os

# 確率を計算するためのクラス
class GaussWishart():
    def __init__(self,dim, mean, var):
        # 事前分布のパラメータ
        self.__dim = dim
        self.__r0 = 1
        self.__nu0 = dim + 2
        self.__m0 = mean.reshape((dim,1))
        self.__S0 = numpy.eye(dim, dim ) * var

        self.__X = numpy.zeros( (dim,1) )
        self.__C = numpy.zeros( (dim, dim) )
        self.__r = self.__r0
        self.__nu = self.__nu0
        self.__N = 0

        self.__update_param()

    def __update_param(self):
        self.__m = (self.__X + self.__r0 * self.__m0)/(self.__r0 + self.__N )
        self.__S = - self.__r * self.__m * self.__m.T + self.__C + self.__S0 + self.__r0 * self.__m0 * self.__m0.T;

    def add_data(self, x ):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X += x
        self.__C += x.dot( x.T )
        self.__r += 1
        self.__nu += 1
        self.__N += 1
        self.__update_param()

    def delete_data(self, x ):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X -= x
        self.__C -= x.dot( x.T )
        self.__r -= 1
        self.__nu -= 1
        self.__N -= 1
        self.__update_param()

    def calc_loglik(self, x):
        def _calc_loglik(self):
            p = - self.__N * self.__dim * 0.5 * math.log( math.pi )
            p+= - self.__dim * 0.5 * math.log( self.__r)
            p+= - self.__nu * 0.5 * math.log( numpy.linalg.det( self.__S ) );

            for d in range(1,self.__dim+1):
                p += math.lgamma( 0.5*(self.__nu+1-d) )

            return p

        # log(P(X))
        p1 = _calc_loglik( self )

        # log(P(x,X))
        self.add_data(x)
        p2 = _calc_loglik( self )
        self.delete_data(x)

        # log(P(x|X) = log(P(x,X)) - log(P(X))
        return p2 - p1

    def get_mean(self):
        return self.__m

    def get_num_data(self):
        return self.__N


def calc_probability( dist, d ):
    return dist.get_num_data() * math.exp( dist.calc_loglik( d ) )


def sample_class( d, distributions, i, bias_dz ):
    K = len(distributions)
    P = [ 0.0 ] * K

    # 累積確率を計算
    P[0] = calc_probability( distributions[0], d ) * bias_dz[i][0]
    for k in range(1,K):
        P[k] = P[k-1] + calc_probability( distributions[k], d ) * bias_dz[i][k]
    
    
    # サンプリング
    rnd = P[K-1] * random.random()
    for k in range(K):
        if P[k] >= rnd:
            return k


def save_result(Pdz, mu, classes, save_dir):
    try:
        os.mkdir( save_dir )
    except:
        pass

    numpy.savetxt( os.path.join( save_dir, "Pdz.txt"),Pdz )
    numpy.savetxt( os.path.join( save_dir, "mu.txt"),mu )
    numpy.savetxt( os.path.join( save_dir, "class.txt"),classes )


# gmmメイン
def train( data, K, num_itr=100, save_dir="model", bias_dz=None, categories=None ):
    # データの次元
    dim = len(data[0])

    Pdz = numpy.zeros((len(data),K))
    mu = numpy.zeros((len(data),dim))

    # データをランダムに分類
    classes = numpy.random.randint( K , size=len(data) )

    # ガウス-ウィシャート分布のパラメータを計算
    mean = numpy.mean( data, axis=0 )
    distributions = [ GaussWishart(dim, mean , 0.1) for _ in range(K) ]
    for i in range(len(data)):
        c = classes[i]
        x = data[i]
        distributions[c].add_data(x)


    for it in range(num_itr):
        # メインの処理
        for i in range(len(data)):
            d = data[i]
            k_old = classes[i]  # 現在のクラス

            # データをクラスから除きパラメータを更新
            distributions[k_old].delete_data( d )
            classes[i] = -1

            # 新たなクラスをサンプリング
            k_new = sample_class( d , distributions, i, bias_dz )

            # サンプリングされたクラスのパラメータを更新
            classes[i] = k_new
            distributions[k_new].add_data( d )

    for m in range(len(data)):
        for n in range(K):
            Pdz[m][n] = calc_probability(distributions[n], d)
            if classes[m] == n:
                mu[m] = distributions[n]._GaussWishart__m.reshape((1,dim))[0]                
                     

    Pdz = (Pdz.T / numpy.sum(Pdz,1)).T
    mu = (mu.T / numpy.sum(mu,1)).T

    save_result(Pdz, mu, classes, save_dir)

    return Pdz, mu

