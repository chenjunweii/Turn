import h5py


def restore(checkpoint, args, nd):
    
    f = h5py.File(checkpoint, 'r')
    
    for k in args.keys():
        
        nd[k] = f[k]

def save(path, nd):
    
    with h5py.File(path) as f:

        for k in nd.keys():

            f.create_dataset(k)
            
            f[k] = nd[k]
