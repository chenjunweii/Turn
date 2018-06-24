import mxnet as mx

def Cross_Entropy(p, g):

    return -(mx.sym.log(mx.sym.maximum(p, 1e-5)) * g + mx.sym.log(mx.sym.maximum(1 - p, 1e-5)) * (1 - g))


def L1(p, g):

    return mx.symbol.abs(p - g)# 0.5 * (p - g
