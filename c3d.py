import cv2
import h5py
import numpy as np
import mxnet as mx

import network
from h import *

class c3d(object):

    def __init__(self, mode, nbatch, size, gpu, model = ''):

        if gpu is True:

            self.device = mx.gpu(0)

        else:

            self.device = mx.cpu()

        self.size = size

        self.nbatch = nbatch

        self.nd = dict()

        self.symbol = dict()

        self.weight = dict()

        self.string = dict()

        self.string['model'] = model;

        self.inf = dict()

        self.shape = dict()

        self.symbol['input'] = mx.symbol.Variable('input')

        self.mode = mode

        if self.mode == 'extract':

            self.build()

            self.initialize()

            self.nd['input'] = mx.nd.zeros(self.shape['input'], self.device)

            self.executor = self.symbol['end'].bind(self.device, self.nd)

    def train(self):

        pass

    def load_pretrained_from_keras(self):

        print('[*] Load Pretrained Model From : {}'.format(self.string['model']))

        m = dict()
        """
        wm['layer_0'] = 'conv1'
        wm['layer_2'] = 'conv2'
        wm['layer_4'] = 'conv3a'
        wm['layer_5'] = 'conv3b'
        wm['layer_7'] = 'conv4a'
        wm['layer_8'] = 'conv4b'
        wm['layer_10'] = 'conv5a'
        wm['layer_11'] = 'conv5b'
        """

        m['conv1'] = 'layer_0'
        m['conv2'] = 'layer_2'
        m['conv3a'] = 'layer_4'
        m['conv3b'] = 'layer_5'
        m['conv4a'] = 'layer_7'
        m['conv4b'] = 'layer_8'
        m['conv5a'] = 'layer_10'
        m['conv5b'] = 'layer_11'
        m['fc6'] = 'layer_15'
        m['fc7'] = 'layer_17'

        #layer_15 : (8192, 4096)
        #layer_17 : (4096, 4096)

        with h5py.File(self.string['model'], 'r') as h5:

            """

            for h in h5.keys():

                try:

                    print('keras => {} : {}'.format(h, h5[h]['param_0'].shape))

                except:

                    pass


            for shape in self.inf['arg']:

                print('shape : {}'.format(shape))

            """

            for arg in self.args:

                if arg != 'input':

                    l = arg.split('_')[0]

                    if arg.endswith('weight'):

                        #print('{} : {}'.format(arg, h5[m[l]]['param_0'].shape))

                        if not arg.startswith('fc'):

                            self.nd[arg] = mx.nd.array(h5[m[l]]['param_0'], self.device)

                        else:

                            self.nd[arg] = mx.nd.array(np.array(h5[m[l]]['param_0']).swapaxes(0,1), self.device)

                    elif arg.endswith('bias'):

                        #print('{} : {}'.format(arg, h5[m[l]]['param_1'].shape))

                        self.nd[arg] = mx.nd.array(h5[m[l]]['param_1'], self.device)

                    else:

                        continue


    def build(self):

        self.shape['input'] = (self.nbatch, 3, 16, self.size.height, self.size.width)

        self.symbol['feature'] = network.c3d(self.symbol['input'], self.weight)

        if self.mode == 'extract':

            self.symbol['end'] = self.symbol['feature']

        self.args = self.symbol['end'].list_arguments()

        self.inf['arg'], self.inf['out'], self.inf['aux'] = self.symbol['end'].infer_shape(**self.shape)

    def inference(self, fb):

        pass

    def initialize(self):

        print(self.string['model'])

        if self.string['model'] != '':

            self.load_pretrained_from_keras()

    def extract(self, fb):

        fb = np.asarray(fb)

        fb = fb.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1).reshape(self.shape['input'])

        self.nd['input'][:] = mx.nd.array(fb, self.device)

        self.executor.forward(False)

        return self.executor.outputs[0].asnumpy()

