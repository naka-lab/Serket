#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

class Module(object):
    __counter = 0
    def __init__(self, name="", learnable=True):
        self.__name = "module%03d_" % Module.__counter + name
        Module.__counter += 1
        self.__forward_prob = None
        self.__backward_prob = None
        self.__learnable = learnable
        self.__observations = None

    def set_forward_msg(self, prob ):
        self.__forward_prob = prob
        
    def get_forward_msg(self):
        return self.__forward_prob

    def get_name(self):
        return self.__name
    
    def connect(self, *obs ):
        self.__observations = obs
        
    def get_observations(self):
        return [ np.array(o.get_forward_msg()) for o in self.__observations ]
    
    def get_backward_msg(self):
        return self.__backward_prob

    def set_backward_msg(self, prob ):
        self.__backward_prob = prob
    
    def send_backward_msgs(self, probs ):
        for i in range(len(self.__observations)):
            self.__observations[i].set_backward_msg( probs[i] )
    
    def update(self):
        pass


class Observation(Module):
    def __init__( self, data, name="obs" ):
        super(Observation,self).__init__( name=name, learnable=False )
        self.set_forward_msg(data)
        