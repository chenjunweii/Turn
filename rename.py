import os

dirs = 'ActivityNet/ActivityNet-1.3'

for v in os.listdir(dirs):

    f = os.path.join(dirs, v)

    nf = os.path.join(dirs, v[2:])

    os.rename(f, nf)

    #print('{} to {}'.format(f, nf))
