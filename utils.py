import os
import h5py
import mxnet as mx


def restore(checkpoint, args, nd, device):
    
    f = h5py.File(checkpoint, 'r')
    
    for k in args:

        if k == 'input' or k == 'groundtruth':

            continue

        nd[k] = mx.nd.array(f[k], device)

    

def save(path, nd):

    if os.path.isfile(path):

        os.remove(path)
    
    with h5py.File(path) as f:

        for k in nd.keys():

            f.create_dataset(k, data = nd[k].asnumpy(), dtype = 'float')
