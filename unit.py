
import os
import cv2
import h5py
import json

import numpy as np

try:

    from . import c3d

except:

    import c3d

def iou(p, f, t = ''):

    """
    p : proposals

    f : foreground

    t : type

    """



    if t == 'clip':

        assert(p[1] > p[0])
        
        try:

            assert(f[1] > f[0])

        except:

            return False

            #raise ValueError('[!] f[1] {} is not greater than f[0] {}'.format(f[1], f[0]))

        f1 = np.copy(f)

        u = p[1] - p[0] # unit size

        if p[0] >= f[0] and p[1] <= f[1]:

            return 1

        elif (p[0] > f[1]):# and p[1] <= f[1]):# or (p[0] > f[0] and p[1] > f[1]):

            return 0

        elif (p[1] < f[0]):# and p[1] > f[1]):

           return 0

        elif (p[1] >= f[0] and p[1] <= f[1]):

            f1[1] = f1[0] + u

        elif (f[0] <= f[1] and p[1] >= f[1]):

            f1[0] = f1[1] - u
        
        union = (min(p[0], f1[0]), max(p[1], f1[1]))
        
        inter = (max(p[0], f1[0]), min(p[1], f1[1]))
        
        iou = float(inter[1] - inter[0]) / (union[1] - union[0])

        return abs(iou)

def sampling(filename, size, unit_size, sample_rate, net, gpu, model = None, dirs = '', path = None, dataset = None, js = None, annotations = None, threshold = None, reuse = False, mode = 'generate'):

    """

    filename : with filename

    mode :

        generate : generating hdf5 file

        inference: return sampled units instead of generating hdf5 file

    """

    h5 = None

    extracted = None

    extracted_id = None

    o = None

    if mode == 'generate':

        if not os.path.isdir(path):

            os.mkdir(path)
        
        o = os.path.join(path, '{}_US[{}]_SR[{}].h5'.format(filename.split('.')[0], unit_size, sample_rate))

        h5 = h5py.File(o)

    elif mode == 'inference':

        extracted = []

        extracted_id = []
        
    fe = None

    if net == 'c3d':

        fe = c3d.c3d('extract', 1, size, gpu, model)

    elif type(net) != str and reuse:

        fe = net

    capture = cv2.VideoCapture(os.path.join(dirs, filename))

    nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(capture.get(cv2.CAP_PROP_FPS))

    fb = []

    fc = []

    fidx = 0 # index of frame
    
    unit = None

    while True:

        ret, frame = capture.read()

        if frame is None:
            
            if mode == 'generate':

                h5.create_dataset('feature_size', data = unit.shape, dtype = 'int32')
                
                h5.create_dataset('sample_rate', data = sample_rate, dtype = 'int32')

                h5.create_dataset('unit_size', data = unit_size, dtype = 'int32')

                h5.create_dataset('nframes', data = nframes, dtype = 'int32')
                
                h5.create_dataset('fps', data = fps, dtype = 'int32')

            #print('findex : {}, nframes : {}'.format(fidx, nframes))
            
            break

        frame = cv2.resize(frame, (size.width, size.height))

        frame = frame - np.array([104, 117, 123])

        if fidx % sample_rate == 0:

            fb.append([])

            fc.append(fidx)

        fb[0].append(frame)

        if len(fb) == 2:

            fb[1].append(frame)

        #print('fb[0] size : {}, fidx : {}, end : {}, fc : {}'.format(len(fb[0]), fidx, fc[0] + unit_size - 1, len(fc)))

        if len(fb[0]) == unit_size:# and fidx == fc[0] + unit_size - 1:

            #print('index : ', fidx)

            unit = fe.extract(fb.pop(0))
            
            start = fc.pop(0)

            d = ('{}_{}'.format(start, start + unit_size - 1))

            if mode == 'generate':
            
                h5.create_dataset(d, data = unit, dtype = 'float')

            elif mode == 'inference':

                extracted.append(unit)

                extracted_id.append([start, start + unit_size - 1])

        fidx += 1
    
    if mode == 'generate':

        print('[*] Unit Level feature of video [ {} ] is save to [ {} ]'.format(os.path.join(dirs, filename), o))
        
    if reuse:

        if dataset == None and mode == 'inference':

            """

            Predict Unit Feature without saving as a hdf5

            """

            return fe, extracted

        if dataset == None:

            """

            Save Unit Feature to hdf5 to predict

            """

            return fe 

        return fe


    elif not reuse:

        # reuse is use for training mulitple video, multiple unit can be extract by calling sampling one time

        if dataset == None and mode == 'inference':

            return extracted, extracted_id


