# -*- coding: utf-8 -*-
"""
Created on Fri May 11 23:11:14 2018

@author: wangz
"""
import numpy as np

class Node(object):
    def __init__(self, value, pro, parent):
        self.value = value
        self.parent = parent
        self.child = []
        self.pro = pro

    def siblings(self):
        if self.parent is None:
            return None
        else:
            return self.parent.child

    def iter_ancestors(self):
        a = self
        while a.parent is not None:
            a = a.parent
            yield a.value

    def ancestors(self):
        a = self
        while a.parent is not None:
            a = a.parent
            yield a.value
        return list(self.iter_ancestors())

    
def train_data(trainingSet):
    [a,b] = np.size(trainingSet)
    T1 = trainingSet[:,1:b-1]
    u = []
    sigma = []
    for i in range(1,b):
        u[i] = np.mean(T1,axis=i)
        sigma[i] = np.std(T1,axis=i)
    return u,sigma

trainingSet = np.random.random((10,8))
[a,b] = np.size(trainingSet)

u, sigma = train_data(trainingSet)
