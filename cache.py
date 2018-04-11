import os
import cv2
import json
import h5py
import ujson
import leveldb
import numpy as np

# custom 

import unit
from h import size

support = [

    'ActivityNet-1.3',

    ]

def generate(dataset, dirs, threshold, size, unit_size, sample_rate, gpu, model, net = '', js = None):

    if dataset not in support:

        raise ValueError('The Dataset {} is not supported right now '.format(dataset))

    annotations = dict()

    annotations['foreground'] = []

    annotations['background'] = []

    db = leveldb.LevelDB(dataset + '_threshold_{}'.format(threshold))
    
    b = leveldb.WriteBatch()

    miss = []

    if dataset == 'ActivityNet-1.3':

        j = json.load(open('ActivityNet/activity_net.v1-3.min.json'))

        for v in j['database'].keys(): # iterate over name of videos

            #print(v)

            vpath = os.path.join(dirs['video'], v + '.mp4') # check if video is downloaded

            if not os.path.isfile(vpath):

                miss.append(v)

                continue

            fpath = os.path.join(dirs['feature'], '{}_US[{}]_SR[{}].h5'.format(v, unit_size, sample_rate)) # check if extracted feature is exist
            
            if not os.path.isfile(fpath):

                # sampling unit level feature

                print('[!] Unit Level Feature [ {} ] is not exist'.format(fpath))

                print('[*] Extracting ... ')

                net = unit.sampling(v + '.mp4', size, unit_size, sample_rate, net, gpu, model, dirs['video'], dirs['feature'], reuse = True)

                #raise ValueError('[!] Unit Level Feature [ {} ] is not exist'.format(fpath))

            else:

                print('Feature Path {} is exist'.format(fpath))
            
            with h5py.File(fpath) as ff:#, cv2.VideoCapture(vpath) as capture:

                #capture = cv2.VideoCapture(vpath)

                #length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                
                #fps = capture.get(cv2.CAP_PROP_FPS)

                #capture.release()

                fps = np.asarray(ff['fps'])
                
                units = list(ff.keys()) # name of units-level feature

                fduration = []

                for fs in range(len(j['database'][v]['annotations'])): # each video contains more than one foreground segment
                    
                    fduration.append((np.asarray(j['database'][v]['annotations'][fs]['segment']) * fps).astype(int)) # frame duration [start frame, end frame]
                for u in units:

                    #print('units : {}'.format(u))

                    if u == 'feature_size' or u == 'unit_size' or u == 'sample_rate' or u == 'nframes' or u == 'fps':

                        continue

                    fg = False

                    #print(len(j['database'][v]['annotations']))

                    for fs in range(len(j['database'][v]['annotations'])): # each video contains more than one foreground segment

                        #print('fs {}'.format(fs))

                        #print(np.asarray(j['database'][v]['annotations'][fs]['segment']) * 8)

                        # convert start time and end time to start frame and end frame respectively

                        """

                        foreground duration : time * fps => index of frame

                        proposal duration : index of frame

                        """
                        
                        #fduration = (np.asarray(j['database'][v]['annotations'][fs]['segment']) * fps).astype(np.int) # frame duration [start frame, end frame]
                        pduration = np.asarray(u.split('_'), dtype = int)

                        iou = unit.iou(pduration, fduration[fs], 'clip')
                        
                        """
                        if iou < 1 and iou > 0:
                            print('Duration Time : {}'.format(j['database'][v]['annotations'][fs]['segment']))
                            print('Duration Time : {}'.format(np.asarray(j['database'][v]['annotations'][fs]['segment']) * fps))
                            print('fduration : {}'.format(fduration[fs]))
                            print('pduration : {}'.format(pduration))
                            print('iou : {}'.format(iou))

                        """
                        if iou > threshold:

                            annotations['foreground'].append('{}_{}'.format(v, u))

                            fg = True

                            break

                    if not fg:

                        annotations['background'].append('{}_{}'.format(v, u))

        #print(ujson.dumps(annotations))

        db.Put('annotations'.encode(), ujson.dumps(annotations).encode())

    db.Write(b, sync = True)

    print('[!] Missing {} Video Files'.format(len(miss)))
    
    print('[!] Without Video Files : ')

    #for v in miss:

    #    print(v)

    #else: 

    #    raise ValueError('The Dataset {} is not supported right now '.format(dataset))

def load(filename):
    
    db = leveldb.LevelDB(filename)

    db = ujson.loads(db.Get('annotations'.encode()).decode())

    print(db)

    print('[*] Keys : {}'.format(db.keys()))

    print('[*] length of foreground : {}'.format(len(db['foreground'])))
    
    print('[*] length of backgroubd : {}'.format(len(db['background'])))

    return db

if __name__ == "__main__":

    load('ActivityNet-1.3_threshold_0.7')
