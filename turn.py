import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

import cache
import unit

from network import *
from data import *
from utils import *
from loss import *
from h import *

class turn(object):

    def __init__(self, config):

        if config.gpu:

            self.device = mx.gpu(config.gpu_id)

        else:

            self.device = mx.cpu()

        self.c = config

        self.symbol = dict()

        self.weight = dict()

        self.nd = dict()
        
        self.inf = dict()

        self.shape = dict()

        self.Executor = None

        self.symbol['input'] = mx.symbol.Variable('input')

        self.symbol['groundtruth'] = mx.symbol.Variable('groundtruth')

    def build(self, mode):

        self.shape['input'] = (self.c.nbatch * (1 + self.c.np_ratio), self.c.unit_feature_size * 3)
        
        self.symbol['pconfidence'], self.symbol['pcoordinate'] = mlp(self.symbol['input'], self.weight, self.c.nhidden, mode)

        if mode == 'train':
            
            self.shape['groundtruth'] = (self.c.nbatch * (1 + self.c.np_ratio), 3)
            
            self.symbol['gconfidence'] = mx.symbol.slice_axis(self.symbol['groundtruth'], axis = 1, begin = 0, end = 1)

            self.symbol['gcoordinate'] = mx.symbol.slice_axis(self.symbol['groundtruth'], axis = 1, begin = 1, end = 3)

            self.symbol['lconfidence'] = mx.symbol.mean(Cross_Entropy(self.symbol['pconfidence'], self.symbol['gconfidence']))
            
            self.symbol['lcoordinate'] = mx.symbol.mean(L1(self.symbol['pcoordinate'],self.symbol['gcoordinate']))
            
            self.symbol['end'] = mx.symbol.MakeLoss(self.symbol['lconfidence'] + self.c.lmda * self.symbol['lcoordinate'])
            
            self.symbol['pconfidence_tit'], self.symbol['pcoordinate_tit'] = mlp(self.symbol['input'], self.weight, self.c.nhidden, 'tit')
            
        elif mode == 'inf':

            self.symbol['end'] = mx.symbol.concat(self.symbol['pconfidence'], self.symbol['pcoordinate'], dim = 1)

        self.args = self.symbol['end'].list_arguments()
        
        self.inf['arg'], self.inf['aux'], self.inf['out'] = self.symbol['end'].infer_shape_partial(**self.shape)
    
    def initialize(self, d, mode, checkpoint):

        if mode == 'train':
            
            initializer = mx.init.Xavier(rnd_type = 'gaussian', factor_type = 'in', magnitude = 2)
            
            self.nd['input'] = mx.nd.array(d.inputs, self.device)
            
            self.nd['groundtruth'] = mx.nd.array(d.targets, self.device)

            self.grad = dict()

            for arg in self.args:

                if arg != 'input' and arg != 'groudtruth':

                    self.grad[arg] = mx.nd.zeros(self.inf['arg'][self.args.index(arg)], self.device)
                    
                    if checkpoint == '':
                    
                        self.nd[arg] = mx.nd.zeros(self.inf['arg'][self.args.index(arg)], self.device)

                        if arg.startswith('w'):

                            initializer._init_weight(arg, self.nd[arg])

                        #initializer(mx.init.InitDesc(arg), self.nd[arg])

                    else:

                        restore(checkpoint, self.args, self.nd, self.device)

            try:

                return int(checkpoint.split('.')[0]) + 1 if checkpoint != '' else 0

            except:

                return 0

    def train(self, epoch, checkpoint = None, se = None, tit = False, ts = None, sl = None, plot = False):

        d = data(self.c, 'train')

        dt = None

        if tit:

            dt  = data(self.c, 'test')

        print('[*] Initialize Training ')

        print('===========================')

        print('       Configuration       ')

        print('===========================')

        print('Batch Size : {}'.format(self.c.nbatch))

        print('Negative / Positive : {}'.format(self.c.np_ratio))

        print('Lambda : {}'.format(self.c.lmda))

        d.next()

        self.build('train')

        init_e = self.initialize(d, 'train', checkpoint)

        e = init_e

        #for i, j in zip(self.args, self.inf['arg']):

            #print('{}, inf => {}, nd => {}'.format(i, j, self.nd[i].shape))

        E = (mx.symbol.Group([self.symbol['end'],
            mx.symbol.MakeLoss(self.symbol['lconfidence']),
            mx.symbol.MakeLoss(self.symbol['lcoordinate'])])).bind(self.device, self.nd, self.grad)

        ET = None

        if tit:

            ET = mx.symbol.abs((self.symbol['pconfidence_tit'] - self.symbol['gconfidence'])).bind(self.device, self.nd)

        lr_scheduler = mx.lr_scheduler.FactorScheduler(self.c.lrds, self.c.lrft)

        adam = mx.optimizer.create('adam', learning_rate = self.c.lr, lr_scheduler = lr_scheduler)

        loss_history = {}
        loss_history['step'] = []
        loss_history['loss'] = []
        
        global_optimal = 0

        while e < epoch + 1:

            E.forward(True)

            E.backward()

            for k in self.args:

                if (k != 'input' and k != 'groundtruth'):
                    
                    idx = self.args.index(k)

                    state = adam.create_state(idx, E.arg_dict[k])

                    adam.update(self.args.index(k), E.arg_dict[k], E.grad_dict[k], state)

            if e % sl == 0 and e != 0:

                print('[*] Step {} => Loss : {:.3f}, Confidence : {:.3f}, Coordinate : {:.3f}, LR : {}'.format(e, E.outputs[0].asnumpy()[0], E.outputs[1].asnumpy()[0], E.outputs[2].asnumpy()[0] * self.c.lmda, lr_scheduler.base_lr))

                if plot:

                    plt.ylabel('Loss')

                    plt.xlabel('Step')

                    plt.ylim([0, 2])

                    loss_history['step'].append(e)

                    loss_history['loss'].append(float(E.outputs[0].asnumpy()[0]))

                    plt.plot(loss_history['step'], loss_history['loss'])
                    
                    #plt.plot(e, E.outputs[0].asnumpy()[0])

                    plt.pause(0.05)
            
            if e % ts == 0 and e != 0 and tit:

                print('\n[*] ===== Testing =====')

                dt.next()
                
                self.nd['input'] = mx.nd.array(dt.inputs, self.device)

                self.nd['groundtruth'] = mx.nd.array(dt.targets, self.device)
                
                ET = mx.symbol.abs((self.symbol['pconfidence'] - self.symbol['gconfidence'])).bind(self.device, self.nd, shared_exec = ET)
                
                ET.forward(False)

                to = ET.outputs[0].asnumpy()

                accuracy = (1 - np.sum(to) / to.shape[0])

                print('[*]  Accuracy : {:.3f}%'.format(accuracy * 100))
                
                if accuracy > global_optimal:

                    global_optimal = accuracy

                    save('optimal_{:.3f}.chk'.format(accuracy * 100), self.nd)

                    print('[*]  Model is save to [ optimal_{:.3f}.chk ]'.format(accuracy * 100))#.format(optimal))

                print('[*] ===================\n')
                
                d.next()

                self.shape['input'] = d.inputs.shape

                self.shape['groundtruth'] = d.targets.shape
                
                self.nd['input'] = mx.nd.array(d.inputs, self.device)

                self.nd['groundtruth'] = mx.nd.array(d.targets, self.device)

                E = (mx.symbol.Group([self.symbol['end'],
                    mx.symbol.MakeLoss(self.symbol['lconfidence']),
                    mx.symbol.MakeLoss(self.symbol['lcoordinate'])])).bind(self.device, self.nd, self.grad, shared_exec = E)
                
            else:

                d.next()

                self.nd['input'][:] = d.inputs #[ = mx.nd.reshape(data = self.nd['input'], shape = d.inputs.shape)

                self.nd['groundtruth'][:] = d.targets

            if e % se == 0 and e != 0:

                save('{}.chk'.format(e), self.nd)

                print('[*] Model is save to [ {}.chk ]'.format(e))


            e += 1

    def test(self):

        #self.nd['input'] = mx.ndarray(

        pass
                
    def inference(self, unit, model, last = False):

        # inference unit not video

        if self.Executor is None:
            
            self.build('inf')

            print('inference')

            restore(model, self.args, self.nd, self.device)

            self.nd['input'] = mx.nd.array(unit, self.device)

            self.Executor = self.symbol['end'].bind(self.device, self.nd)
        
        elif last:

            self.shape['input'] = unit.shape

            self.nd['input'] = mx.nd.array(unit, self.device)
            
            self.Executor = self.Executor.reshape(allow_up_sizing = False, **self.shape)
            
            self.Executor = self.symbol['end'].bind(self.device, self.nd, shared_exec = self.Executor)

        else:

            self.nd['input'][:] = mx.nd.array(unit, self.device)

        self.Executor.forward(False)

        return self.Executor.outputs[0].asnumpy()

    def generate_foreground_video(self, video, mfe, mpp):

        unit_feature, unit_feature_id = unit.sampling(video, size(self.c.size), self.c.unit_size, self.c.sample_rate, self.c.net, self.c.gpu, model = mfe, mode = 'inference')  

        uidx = 0

        ibatch = None # inference batch

        confidence = np.zeros([0, 3])

        l = len(unit_feature_id)

        iii = 0

        while True:
            
            ibatch = self.load_unit_feature_with_context_inference_batch(uidx, unit_feature_id, unit_feature)

            confidence = np.vstack((confidence, self.inference(ibatch, mpp)))

            uidx += self.c.nbatch

            if uidx + self.c.nbatch == l or uidx > l:

                break

            elif uidx + self.c.nbatch > l:

                ibatch = self.load_unit_feature_with_context_inference_batch(uidx, unit_feature_id, unit_feature)

                confidence = np.vstack((confidence, self.inference(ibatch, mpp, last = True)))

                break
        
        foreground = []

        with_context = False

        for uidx in range(len(confidence)):

            if confidence[uidx, 0] > self.c.threshold:

                if with_context:

                    if uidx <= self.c.nctx:

                        foreground += list(range(0, (uidx + self.c.nctx) * self.sample_rate + self.c.unit_size))

                    elif uidx + self.c.nctx > len(confidence):

                        foreground += list(range((uidx - self.c.nctx) * self.c.sample_rate, len(confidence) * self.c.sample_rate + self.c.unit_size))

                    else:

                        foreground += list(range((uidx - self.c.nctx) * self.c.sample_rate, (uidx + self.c.nctx) * self.c.sample_rate + self.c.unit_size))

                else:

                    print('uidx : ', uidx)
                
                    foreground += list(range(uidx * self.c.sample_rate, uidx * self.c.sample_rate + self.c.unit_size))

        
        foreground = set(foreground)

        print(foreground)
        
        if len(foreground) > 0:
            
            capture = cv2.VideoCapture(video)

            nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            fps = int(capture.get(cv2.CAP_PROP_FPS))

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
           
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print('Total Frames extracted from unit : ', len(foreground))

            print('Number of frames in video : ', nframes)

            foreground_frames = []

            i = 0

            while True:
                 
                ret, frame = capture.read()

                if frame is None:

                    break

                elif i in foreground:

                    foreground_frames.append(frame)
                
                i += 1
            
            codec = cv2.VideoWriter_fourcc(*'MJPG')

            writer = cv2.VideoWriter('{}_action_unit.mkv'.format(video), codec, fps, (width, height))

            for f in foreground_frames:

                writer.write(f)

            writer.release()

            capture.release()

        else:

            print('[!] No Foreground Unit')


    def load_unit_feature_with_context_inference_batch(self, buidx, unit_feature_id, unit_feature):

        # buidx => index of first unit in batch

        unit_feature_with_context = np.zeros([0, 3 * self.c.unit_feature_size])

        for b in range(self.c.nbatch):

            uidx = buidx + b # focus on current unit in batch by adding offset

            if uidx >= len(unit_feature_id):

                break

            anchor_feature = unit_feature[uidx]

            prev_feature = None

            next_feature = None

            #print('uidx : ', uidx)

            if uidx - self.c.nctx < 0 or uidx + self.c.nctx > len(unit_feature_id):

                if uidx - self.c.nctx < 0:

                    if uidx == 0:

                        prev_feature = np.zeros([self.c.nctx, self.c.unit_feature_size])

                    else:
                        
                        padding = np.zeros([self.c.nctx - uidx, self.c.unit_feature_size])

                        prev_feature = np.asarray(unit_feature[ : uidx])

                        prev_feature = np.vstack([padding, prev_feature.reshape([prev_feature.shape[0], -1])])
                
                    next_feature = np.asarray(unit_feature[uidx : uidx + self.c.nctx]).reshape([self.c.nctx, -1])

                elif uidx + self.c.nctx > len(unit_feature_id):

                    if uidx == len(unit_feature_id):

                        next_feature = np.zeros([self.c.nctx, self.c.unit_feature_size])

                    else:

                        padding = np.zeros([self.c.nctx + uidx - len(unit_feature_id), self.c.unit_feature_size])

                        next_feature = np.asarray(unit_feature[uidx : ])

                        next_feature = np.vstack([next_feature.reshape([next_feature.shape[0], -1]), padding])

                    prev_feature = np.asarray(unit_feature[uidx - self.c.nctx : uidx]).reshape([self.c.nctx, -1])

            else:

                next_feature = np.asarray(unit_feature[uidx : uidx + self.c.nctx]).reshape([self.c.nctx, -1])

                prev_feature = np.asarray(unit_feature[uidx - self.c.nctx : uidx]).reshape([self.c.nctx, -1])
    
            anchor_feature = anchor_feature.reshape([-1])

            next_feature = np.mean(next_feature, axis = 0)

            prev_feature = np.mean(prev_feature, axis = 0)

            # vstack over batch

            unit_feature_with_context = np.vstack([unit_feature_with_context,
                    np.concatenate([prev_feature, anchor_feature, next_feature]).reshape([1, -1])])
        
        return unit_feature_with_context
                
