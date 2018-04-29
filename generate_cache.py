import os
import h5py
import argparse

import cache

from h import size

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video', default = '') 

parser.add_argument('-m', '--model', default = 'c3d-sports1M_weights.h5')

parser.add_argument('-n', '--net', default = 'c3d')

parser.add_argument('-o', '--output', default = 'extract')

parser.add_argument('-u', '--unit_size', type = int, default = 16)

parser.add_argument('-s', '--sample_rate', type = int, default = 8)

parser.add_argument('-f', '--force', action = 'store_true', default = False)

parser.add_argument('-gpu', action = 'store_true', default = False)

parser.add_argument('-ds', '--dataset', default = None)

parser.add_argument('-th', '--threshold', type = float, default = 0.7)

parser.add_argument('-sz', '--size', default = [224,224])

parser.add_argument('-vd', default = '')

parser.add_argument('-fd', default = '')

parser.add_argument('-js', default = '')

parser.add_argument('-tt', type = float, default = 3)

args = parser.parse_args()

dirs = dict()

dirs['video'] = args.vd

dirs['feature'] = args.fd

cache.generate(args.dataset, dirs, args.threshold, size(args.size), args.unit_size, args.sample_rate, args.gpu, args.model, args.force, args.tt, args.net, js = None)

