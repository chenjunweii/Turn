import os
import cv2
import json
import h5py
import leveldb
import numpy as np


import cache

class data(object):

    """
    
    WorkFlow

    __init__ => load_cache

    next => next_batch => next_annotation => load_unit_feature_with_context => load_unit_feature

    """

    class ann(object):

        start = None

        end = None

        video = None

    def __init__(self, c):

        self.c = c

        self.step = 0

        self.feature = dict()

        self.annotations = dict()

        self.feature['foreground'] = []

        self.feature['background'] = []
        
        self.annotations['foreground'] = []

        self.annotations['background'] = []

        self.load_cache()

    def load_cache(self):

        self.cache = cache.load('{}_threshold_{}'.format(self.c.dataset, self.c.threshold))

        self.p_size = len(self.cache['foreground'])

        self.n_size = len(self.cache['background'])

        self.p_shuffle_counter = self.p_size

        self.n_shuffle_counter = self.n_size
        
        self.p_virtual = range(self.p_size)
        
        self.p_physical = range(self.p_size)
        
        self.n_virtual = range(self.n_size)

        self.n_physical = range(self.n_size)

    def load_annotations(self, h5f, video, start, end):

        return []

    def next_batch_data(self):

        # call at each batch

        #self.length = dict()
        
        #self.length['foreground'] = None

        #self.length['background'] = None

        #with open(self.config.fg_list) as f:

        """
        
        concat 

        """

 
        for p in self.p_batch:

            v, s, e = self.cache['foreground'][self.p_physical[self.p_virtual[p]]].split('_')
                    
            fpath = os.path.join(self.c.dirs['feature'], '{}_US[{}]_SR[{}].h5'.format(v, self.c.unit_size, self.c.sample_rate))
 
            if os.path.exists(fpath):
                
                with h5py.File(fpath) as h5f:

                    self.feature['foreground'].append(self.load_unit_feature_with_context(h5f, v, s, e)) # load feature of foreground units

                    self.annotations['foreground'].append(self.load_annotations(h5f, v, s, e, 1))

            else:

                raise RuntimeError('[!] No Such Feature File [ {} ]'.format(fpath))
        
        for n in self.n_batch:

            v, s, e = self.cache['background'][self.n_physical[self.n_virtual[n]]].split('_')

            fpath = os.path.join(self.c.dirs['feature'], '{}_US[{}]_SR[{}].h5'.format(v, self.c.unit_size, self.c.sample_rate))
            
            if os.path.exists(fpath):
                
                with h5py.File(fpath) as h5f:
            
                    self.feature['background'].append(self.load_unit_feature_with_context(h5f, v, s, e)) # load feature of background units

                    self.annotations['background'].append(self.load_annotations(h5f, v, s, e, 1))
        
            else:

                raise RuntimeError('[!] No Such Feature File [ {} ]'.format(fpath))
    
    def next(self):
        
        self.inputs = []

        self.targets = []

        self.next_batch() # choose index of sample in dataset for positive sample and negative sample

        self.next_batch_data() # load the feature file and annotation of the positive sample and negative sample that we just get by calling function 'self.next_batch'

            
    def next_batch(self):

        begin = self.step * self.c.nbatch % self.p_size

        end = (self.step + 1) * self.c.nbatch % self.p_size

        if (begin + self.c.nbatch) > self.p_size:

            self.p_batch = self.p_virtual[ begin : ] + self.p_virtual[ : end ]

        elif (begin + self.c.nbatch) == self.p_size:

            self.p_batch = self.p_virtual[ begin : ]

        else:

            self.p_batch = self.p_virtual[ begin : end ]

        begin = self.step * (self.c.nbatch * self.c.np_ratio) % self.n_size

        end = (self.step + 1) * self.c.nbatch % self.n_size

        if (begin + self.c.nbatch) > self.n_size:

            self.n_batch = self.n_virtual[ begin : ] + self.n_virtual[ : end ]

        elif (begin + self.c.nbatch) == self.n_size:

            self.n_batch = self.n_virtual[ begin : ]

        else:

            self.n_batch = self.n_virtual[ begin : end ]

        self.step += 1;

        self.p_shuffle_counter -= self.c.nbatch;

        self.n_shuffle_counter -= self.c.nbatch * self.c.np_ratio;

        if self.p_shuffle_counter <= 0:

            random.shuffle(self.p_physical);

            self.p_shuffle_counter = self.p_size

        if self.n_shuffle_counter <= 0:

            random.shuffle(self.n_physical);

            self.n_shuffle_counter = self.n_size

    def load_unit_feature_with_context(self, h5f, video, start, end):

        start = int(start)

        print('Start : ', start)

        end = int(end)

        print('End : ', end) 
        
        feature_next = self.load_unit_feature('next', h5f, start, end)

        feature_anchor = self.load_unit_feature('anchor', h5f, start, end)

        feature_prev = self.load_unit_feature('prev', h5f, start, end)

        print('next : ', feature_next.shape)

        print('anchor : ', feature_anchor.shape)

        print('prev : ', feature_prev.shape)

        feature = np.concatenate([feature_prev, feature_anchor, feature_next])



    def load_unit_feature(self, mode, h5f, start, end):

        frame_index = 0

        print('mode : ', mode)

        if mode == 'anchor':
            
            frame_index = start

        elif  mode == 'next':

            frame_index = end + 1

        elif mode == 'prev':

            frame_index = start -  self.c.unit_size * self.c.nctx

        offset = 0

        feature = np.zeros([0, self.c.unit_feature_size])

        nframes = np.asarray(h5f['nframes'])

        if frame_index < 0:

            frame_index = 0

        elif frame_index >= nframes:

            return np.zeros(self.c.unit_feature_size)

        # Context Unit offset
        
        if mode == 'next' or mode == 'prev':

            while offset < self.c.nctx: 
                
                print('nctx : ', self.c.nctx)
                
                frame_start = frame_index

                frame_end = frame_index + self.c.unit_size - 1

                print('frame idx : ', frame_index)
                
                print('frame end : ', frame_end)

                if frame_start < 0 or frame_end > nframes:

                    feature = np.vstack((feature, np.zeros(self.c.unit_feature_size)))

                else:

                    u = '{}_{}'.format(frame_start, frame_end)

                    feature = np.vstack((feature, np.asarray(h5f[u])))

                frame_index += self.c.unit_size
                
                offset += 1
            
            pooled = np.mean(feature, axis = 0) # mean pooling if it is positive sample

            return pooled


        elif mode == 'anchor':

            frame_start = frame_index

            frame_end = frame_index + self.c.unit_size - 1

            print('frame idx : ', frame_index)
            
            print('frame end : ', frame_end)

            u = '{}_{}'.format(frame_start, frame_end)
            
            print('h5f[u] shape ', h5f[u].shape)

            feature = np.vstack((feature, np.asarray(h5f[u])))

            print(feature.shape)

            frame_index += self.c.unit_size

            pooled = np.mean(feature, axis = 0) # mean pooling if it is positive sample

            return pooled


