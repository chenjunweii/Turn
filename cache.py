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

def generate(dataset, dirs, threshold, size, unit_size, sample_rate, gpu, model, force, tt, net = '', js = None):

    if dataset not in support:

        raise ValueError('The Dataset {} is not supported right now '.format(dataset))

    dictionary = dict()

    dictionary['train'] = dict()

    dictionary['train']['foreground'] = []

    dictionary['train']['background'] = []

    dictionary['train']['annotations'] = dict()
    
    dictionary['train']['response'] = dict()

    dictionary['train']['iou'] = dict()

    db = dict()

    db['train'] = leveldb.LevelDB(dataset + '_threshold_{}_train'.format(threshold))
    
    if tt > 1:

        db['test'] = leveldb.LevelDB(dataset + '_threshold_{}_test'.format(threshold))
        
        dictionary['test'] = dict()

        dictionary['test']['foreground'] = []

        dictionary['test']['background'] = []

        dictionary['test']['annotations'] = dict()
        
        dictionary['test']['response'] = dict()

        dictionary['test']['iou'] = dict()

    b = leveldb.WriteBatch()

    miss = []

    fg_counter = 0

    bg_counter = 0

    if dataset == 'ActivityNet-1.3':

        j = json.load(open('ActivityNet/activity_net.v1-3.min.json'))

        for v in j['database'].keys(): # iterate over name of videos

            vpath = os.path.join(dirs['video'], v + '.mp4') # check if video is downloaded

            if not os.path.isfile(vpath):

                miss.append(v)

                continue

            fpath = os.path.join(dirs['feature'], '{}_US[{}]_SR[{}].h5'.format(v, unit_size, sample_rate)) # check if extracted feature is exist
            
            exist = os.path.isfile(fpath)

            if not exist and not force:

                # sampling unit level feature

                print('[!] Unit Level Feature [ {} ] is not exist'.format(fpath))

                print('[*] Extracting ... ')

                net = unit.sampling(v + '.mp4', size, unit_size, sample_rate, net, gpu, model, dirs['video'], dirs['feature'], reuse = True)

                #raise ValueError('[!] Unit Level Feature [ {} ] is not exist'.format(fpath))

            elif exist:

                print('Feature Path {} is exist'.format(fpath))
            
            elif force:

                print('[!] Unit Level Feature [ {} ] is not exist, Ignoring...'.format(fpath))

            if not force or (force and exist) :
            
                with h5py.File(fpath) as ff:#, cv2.VideoCapture(vpath) as capture:

                    #capture = cv2.VideoCapture(vpath)

                    #length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    
                    #fps = capture.get(cv2.CAP_PROP_FPS)

                    #capture.release()

                    print('[*] Current Video => [ {} ]'.format(fpath))

                    fps = np.asarray(ff['fps'])
                    
                    units = list(ff.keys()) # name of units-level feature

                    fduration = []
                    
                    #annotations['annotations'][v] = []


                    for fs in range(len(j['database'][v]['annotations'])): # each video contains more than one foreground segment
                        
                        fduration.append((np.asarray(j['database'][v]['annotations'][fs]['segment']) * fps).astype(int)) # frame duration [start frame, end frame]
                        #annotations['annotations'][v].append(fduration[fs])

                        print(fduration[fs])

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

                            if not iou:

                                break
                            
                            """
                            if iou < 1 and iou > 0:
                                print('Duration Time : {}'.format(j['database'][v]['annotations'][fs]['segment']))
                                print('Duration Time : {}'.format(np.asarray(j['database'][v]['annotations'][fs]['segment']) * fps))
                                print('fduration : {}'.format(fduration[fs]))
                                print('pduration : {}'.format(pduration))
                                print('iou : {}'.format(iou))

                            """
                            if iou > threshold:

                                vu = '{}_{}'.format(v, u)

                                if fg_counter >= tt:

                                    dictionary['test']['foreground'].append(vu)

                                    dictionary['test']['response'][vu] = '{}_{}'.format(fduration[fs][0], fduration[fs][1])

                                    dictionary['test']['iou'][vu] = iou

                                    fg_counter = 0
                                    
                                else:

                                    dictionary['train']['foreground'].append(vu)

                                    dictionary['train']['response'][vu] = '{}_{}'.format(fduration[fs][0], fduration[fs][1])

                                    dictionary['train']['iou'][vu] = iou

                                    fg_counter += 1

                                fg = True

                                break

                        if not fg:

                            if bg_counter >= tt:

                                dictionary['test']['background'].append('{}_{}'.format(v, u))

                                bg_counter = 0

                            else:

                                dictionary['train']['background'].append('{}_{}'.format(v, u))

                                bg_counter += 1

        #print(ujson.dumps(annotations))

        db['train'].Put('annotations'.encode(), ujson.dumps(dictionary['train']).encode())

        if tt > 1:

            db['test'].Put('annotations'.encode(), ujson.dumps(dictionary['test']).encode())

    db['train'].Write(b, sync = True)
    
    print('[*] Cache is save to [ {} ]'.format(dataset + '_threshold_{}_train'.format(threshold)))
    
    if tt > 1:
    
        db['test'].Write(b, sync = True)
        
        print('[*] Cache is save to [ {} ]'.format(dataset + '_threshold_{}_test'.format(threshold)))

    print('[!] Missing {} Video Files'.format(len(miss)))
    
    #print('[!] Without Video Files : ')

    #for v in miss:

    #    print(v)

    #else: 

    #    raise ValueError('The Dataset {} is not supported right now '.format(dataset))

def load(filename, mode = ''):
    
    db = leveldb.LevelDB(filename)

    db = ujson.loads(db.Get('annotations'.encode()).decode())

    #print('[*] Keys : {}'.format(db.keys()))

    if mode == '':

        print('[*] Number of Foreground Sample : {}'.format(len(db['foreground'])))
        
        print('[*] Number of Background Sample : {}'.format(len(db['background'])))

    else:


        print('[*] Number of Foreground Sample For {} : {}'.format(mode, len(db['foreground'])))
        
        print('[*] Number of Background Sample For {} : {}'.format(mode, len(db['background'])))

    return db

if __name__ == "__main__":

    load('ActivityNet-1.3_threshold_0.7')
