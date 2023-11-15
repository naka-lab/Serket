#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys

sys.path.append("../")

import serket as srk
import numpy as np
import os
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class TorchNN(srk.Module, metaclass=ABCMeta):
    def __init__(
        self,
        batch_size1,
        batch_size2,
        itr1=5000,
        itr2=5000,
        lr=1e-3,
        device="cpu",
        name="nn",
        load_dir=None,
    ):
        super(TorchNN, self).__init__(name, True)
        self.__itr1 = itr1
        self.__itr2 = itr2
        self.__batch_size1 = batch_size1
        self.__batch_size2 = batch_size2
        self.__load_dir = load_dir
        self.__device = device
        self.__n = 0
        self.__lr = lr

    def tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.__device)

    def numpy(self, x):
        return x.cpu().detach().numpy()

    # クラス継承先でmodelを定義
    @abstractmethod
    def build_model(self, input_dim, output_dim):
        pass

    def train(self):
        softmax = nn.Softmax()
        loss_func = nn.MSELoss()
        w = torch.ones(self.__data[0].shape, requires_grad=True)

        # 順方向学習
        model = self.build_model(self.__input_dim, self.__output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.__lr)
        for epoch in range(1, self.__itr1 + 1):
            batch_size = self.__batch_size1
            sff_idx = np.random.permutation(self.__N)

            for idx in range(0, self.__N, batch_size):
                in_data = self.__data[0][
                    sff_idx[
                        idx : idx + batch_size
                        if idx + batch_size < self.__N
                        else self.__N
                    ]
                ]
                out_data = self.__data[1][
                    sff_idx[
                        idx : idx + batch_size
                        if idx + batch_size < self.__N
                        else self.__N
                    ]
                ]

                # pattern 1
                predicts = model(in_data)

                # pattern 2: こちらはうまういかない
                # predicts = model( in_data * 1/self.__N )

                loss = loss_func(out_data, predicts)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 逆方向
        optimizer = torch.optim.Adam([w], lr=self.__lr)
        for epoch in range(1, self.__itr2 + 1):
            in_data = self.__data[0]
            out_data = self.__data[1]

            q = softmax(w)
            qp = q * in_data
            predicts = model(qp)
            loss = loss_func(out_data, predicts) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        message = self.numpy(softmax(w))

        torch.save(model.state_dict(), os.path.join(self.__save_dir, "model.pth"))
        np.savetxt(os.path.join(self.__save_dir, "message.txt"), message)
        np.savetxt(os.path.join(self.__save_dir, "weight.txt"), self.numpy(w))

        return message

    def update(self):
        self.__data = self.get_observations()
        self.__data[0] = self.tensor(self.__data[0])
        self.__data[1] = self.tensor(self.__data[1])

        self.__save_dir = os.path.join(self.get_name(), "%03d" % self.__n)
        if not os.path.exists(self.__save_dir):
            os.makedirs(self.__save_dir)

        self.__N = len(self.__data[0])  # データ数
        self.__input_dim = len(self.__data[0][0])  # 入力の次元数
        self.__output_dim = len(self.__data[1][0])  # 出力の次元数

        if self.__load_dir is None:
            save_dir = os.path.join(self.get_name(), "%03d" % self.__n)
        else:
            save_dir = os.path.join(self.get_name(), "recog")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # NN学習
        message = self.train()

        self.__n += 1

        # メッセージの送信
        self.send_backward_msgs([message, None])

