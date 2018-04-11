import argparse

from turn import turn
from config import config


parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', action = 'store_true', default = False)

parser.add_argument('-i', '--inference', action = 'store_true', default = False)

parser.add_argument('-v', '--video', default = '') 

parser.add_argument('-ds', '--dataset', default = '') 

parser.add_argument('-u', '--unit', default = '') 

parser.add_argument('-m', '--model', default = '')

parser.add_argument('-c', '--checkpoint', default = '')

parser.add_argument('-e', '--epoch', type = int, default = 1000)

parser.add_argument('-s', '--save', type = int, default = 100)

parser.add_argument('-lr', type = float, default = 0.0001)

parser.add_argument('-fd', type = str, default = '')

parser.add_argument('-vd', type = str, default = '')

parser.add_argument('-gpu', action = 'store_true', default = False)

args = parser.parse_args()

c = config()

"""
configuation

"""


c.nbatch = 1

c.gpu = args.gpu 

c.lr = args.lr

c.nhidden = 1000

c.nctx = 5

c.dataset = args.dataset

c.dirs['feature'] = args.fd

c.dirs['video'] = args.vd

#c.checkpoint_step

"""

creating turn instance

"""
print('[*] Creating TURN Instance...')

if c.dataset == '':

    raise ValueError('[!] Dataset is not specified correctly')

t = turn(c)

if args.train:

    t.train(args.epoch, checkpoint = args.checkpoint, se = args.save)

elif args.inference != '' and args.video != '':

    t.inference(args.video, args.model)

elif args.inference != '' and args.unit != '':

    t.inference(args.unit, args.model)

