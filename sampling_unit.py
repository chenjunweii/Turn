import os
import h5py
import argparse

# custom

import unit
from h import size

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video', default = '') 

parser.add_argument('-d', '--dir', default = '') 

parser.add_argument('-m', '--model', default = 'c3d-sports1M_weights.h5')

parser.add_argument('-n', '--net', default = 'c3d')

parser.add_argument('-o', '--output', default = 'extract')

parser.add_argument('-u', '--unit_size', type = int, default = 16)

parser.add_argument('-s', '--sample_rate', type = int, default = 8)

parser.add_argument('-gpu', action = 'store_true', default = False)

parser.add_argument('-ds', '--dataset', default = None)

parser.add_argument('-th', '--threshold', default = None)

parser.add_argument('-sz', '--size', default = [224, 224])

args = parser.parse_args()

if args.video != '' and args.dir == '':

    print('[*] Overlap Area : {0:.0f}%'.format((args.unit_size - args.sample_rate) / args.unit_size * 100))

    unit.sampling(args.video, size(args.size), args.unit_size, args.sample_rate, args.net, args.gpu, args.model, args.dir, args.output)

elif args.video == '' and args.dir != '':
    
    print('[*] Overlap Area : {0:.0f}%'.format((args.unit_size - args.sample_rate) / args.unit_size * 100))

    if os.path.isdir(args.dir):

        for v in os.listdir(args.dir):

            if 'mp4' in v or 'mpg' in v or 'mkv' in v or 'avi' in v:
   
                args.net = unit.sampling(v, size(args.size), args.unit_size, args.sample_rate, args.net, args.gpu, args.model, args.dir, args.output, reuse = True)

    else:

        raise ValueError('Directory is not exists')

else:
    
    raise ValueError('Parameters are not setting properly')
