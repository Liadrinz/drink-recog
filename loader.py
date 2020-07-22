import json

import numpy as np

from random import shuffle
from config import conf

class Loader:
    
    def __init__(self):
        trains = self.load_dataset('train')
        tests = self.load_dataset('test')
        self.trainX = trains['images']
        self.trainY = trains['tags']
        self.testX = tests['images']
        self.testY = tests['tags']
        self.train_l = len(self.trainX)
        self.test_l = len(self.testX)
        self._train_ptr = 0
        self._test_ptr = 0
    
    def fetch(self, size, which):
        X = []
        Y = []
        if which == 'train':
            left = self._train_ptr
            l = self.train_l
        else:
            left = self._test_ptr
            l = self.test_l
        if which == 'train':
            for d in range(size):
                X.append(self.trainX[(left + d) % l])
                Y.append(self.trainY[(left + d) % l])
        elif which == 'test':
            for d in range(size):
                X.append(self.trainX[(left + d) % l])
                Y.append(self.trainY[(left + d) % l])
        return np.array(X), np.array(Y)
            
    def incre(self, size, which):
        if which == 'train':
            self._train_ptr += size
            self._train_ptr %= self.train_l
        elif which == 'test':
            self._test_ptr += size
            self._test_ptr %= self.test_l

    def load_dataset(self, which):
        with open(conf(f'aug.{which}.output'), 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        return data

    def next_batch(self, size, which):
        X1, Y1 = self.fetch(size, which)
        self.incre(size, which)
        return X1, Y1