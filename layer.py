
def fc(arg, inputs, weights, nhidden):
    
    weights['w_' + arg] = mx.symbol.Variable('w_' + arg)

    weights['b_' + arg] = mx.symbol.Variable('b_' + arg)
    
    return mx.sym.FullyConnected(fc1_dropout, weights['w_' + arg], weights['b_' + arg], nhidden)

def conv(arg, inputs, weights):

    pass


def deconv(arg, inputs, weights):

    pass


