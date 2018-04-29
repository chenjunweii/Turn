import argparse

from turn import turn
from config import config


parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', action = 'store_true', default = False)

parser.add_argument('-i', '--inference', action = 'store_true', default = False)

parser.add_argument('-g', '--generate', action = 'store_true', default = False)

parser.add_argument('-v', '--video', default = '') 

parser.add_argument('-ds', '--dataset', default = '') 

parser.add_argument('-u', '--unit', default = '') 

parser.add_argument('-m', '--model', default = '')

parser.add_argument('-mfe', default = '')

parser.add_argument('-mpp', default = '')

parser.add_argument('-c', '--checkpoint', default = '')

parser.add_argument('-e', '--epoch', type = int, default = 1000)

parser.add_argument('-te', type = int, default = 0)

parser.add_argument('-tit', action = 'store_true', default = False)

parser.add_argument('-s', '--save', type = int, default = 500)

parser.add_argument('-sl', type = int, default = 100) # show loss

parser.add_argument('-np', type = int, default = 3)

parser.add_argument('-b', type = int, default = 64)

parser.add_argument('-tb', type = int, default = 1024)

parser.add_argument('-lrft', type = float, default = 0.95)

parser.add_argument('-lrds', type = int, default = 2500)

parser.add_argument('-lr', type = float, default = 0.0001)

parser.add_argument('-th', type = float, default = 0.7)

parser.add_argument('-lmda', type = float, default = 0.0005)

parser.add_argument('-fd', type = str, default = '')

parser.add_argument('-vd', type = str, default = '')

parser.add_argument('-gpu', action = 'store_true', default = False)

parser.add_argument('-gid', type = int, default = 0)

parser.add_argument('-dbg', action = 'store_true', default = False)

parser.add_argument('-plot', action = 'store_true', default = False)

args = parser.parse_args()

c = config()

"""
configuation

"""


c.nbatch = args.b

c.tnbatch = args.tb

c.gpu = args.gpu 

c.gpu_id = args.gid

c.lr = args.lr

c.lrds = args.lrds

c.lrft = args.lrft

c.nhidden = 1000

c.nctx = 5

c.dataset = args.dataset

c.dirs['feature'] = args.fd

c.dirs['video'] = args.vd

#c.checkpoint_step

c.net = 'c3d'

c.np_ratio = args.np

c.size = [224, 224]

c.lmda = args.lmda

c.threshold = args.th

c.debug = args.dbg

"""

creating turn instance

"""
print('[*] Creating TURN Instance...')


t = turn(c)

if args.train:
    
    if c.dataset == '':

        raise ValueError('[!] Dataset is not specified correctly')

    t.train(args.epoch, checkpoint = args.checkpoint, se = args.save, tit = args.tit, ts = args.te, sl = args.sl, plot = args.plot)

elif args.inference and args.video != '':

    t.inference(args.video, args.model)

elif args.inference and args.unit != '' and args.video == '':

    t.inference(args.unit, args.model)

elif args.generate and args.video != '':

    t.generate_foreground_video(args.video, args.mfe, args.mpp)

