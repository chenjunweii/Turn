import mxnet as mx
import numpy as np
import h5py


import cache

from network import *
from data import *
from utils import *



class turn(object):

    def __init__(self, config):

        if config.gpu:

            self.device = mx.gpu()

        else:

            self.device = mx.cpu()

        self.config = config

        self.symbol = dict()

        self.weight = dict()

        self.nd = dict()
        
        self.inf = dict()

        self.shape = dict()

        self.symbol['input'] = mx.symbol.Variable('input')

        self.symbol['groundtruth'] = mx.symbol.Variable('groundtruth')

    def build(self, mode):

        self.shape['input'] = (self.nbatch, )
        
        self.symbol['prediction'] = mlp(self.symbol['input'], self.weight, self.nhidden)
        
        self.args = self.symbol['objective'].list_arguments()
        
        if mode == 'train':

            self.symbol['objective'] = MakeLoss(crossentropy(self.symbol['prediction'],  self.symbol['groundtruth']))

            self.symbol['end'] = self.symbol['objective']
        
            self.shape['groundtruth'] = []

        elif mode == 'inf':

            self.symbol['end'] = self.symbol['prediction']

        self.inf['arg'], self.inf['aux'], self.inf['out'] = self.symbol['end'].InferenceShape(**self.shape)
    
    def initialize(self, d):

        printf('[*] Initialize')

        initializer = mx.init.Xavier(rnd_type = 'gaussian', factor_type = 'in', magnitude = 2)

        self.nd['input'] = d.inputs

        self.nd['groundtruth'] = d.targets

        if checkpoint == None:

            for arg in self.inf['arg'].keys():

                if arg != 'input' and arg != 'groudtruth':

                    self.nd[arg] = mx.zeros(self.inf[arg])

                    if arg.startswith('w'):

                        #initializer.init(self.nd[arg])

                        initializer(mx.init.InitDesc(arg), self.nd[arg])

        else:

            restore(checkpoint, self.args, self.nd)

    def train(self, epoch, checkpoint = None, se = None):

        print('import data')

        d = data(self.config)

        print('train')

        d.next()

        self.build('train')

        self.initialize(d, checkpoint)

        E = self.symbol['objective'].bind(mx.gpu(), self.nd)

        adam = mx.optimizer.create('adam', learning_rate = learning_rate)

        for e in range(epoch):

            E.forward(True)

            E.backward()

            for k in arg.keys():

                if k != 'input' and k != 'grountruth':

                    adam.update(E.arg_dict[k], E.grad_dict[k])

            print('loss : {}'.format(E.outputs[0].asnumpy()))

            d.next()

            self.nd['input'][:] = d.inputs

            self.nd['groundtruth'][:] = d.targets

            if e % se == 0 and e != 0:
            
               save('{}.chk'.format(e))
                
    def inference(self, unit, model):

        # inference unit not video

        self.build('inf')

        self.args = self.symbol['end'].list_arguments()
        
        restore(model, self.args, self.nd)

        E = self.symbol['objective'].bind(mx.gpu(), self.nd)

        E.forward()


