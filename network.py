import mxnet as mx

def fc(name, inputs, weights, nhidden, mode = ''):

    if mode == '':

        weights['w_' + name] = mx.sym.Variable('w_' + name)

        weights['b_' + name] = mx.sym.Variable('b_' + name)

        return mx.sym.FullyConnected(inputs, weights['w_' + name], weights['b_' + name], nhidden)

    elif mode == 'tit':

        return mx.sym.FullyConnected(inputs, weights['w_' + name], weights['b_' + name], nhidden, name = name + '_' + mode)

def mlp(inputs, weights, nhidden, mode, dropout = 0.5):

    fc1 = fc('fc1', inputs, weights, nhidden)
    
    fc1_relu = mx.sym.Activation(fc1, 'relu')

    outputs = None

    if mode == 'train':

        fc1_dropout = mx.sym.Dropout(fc1_relu, dropout)

        outputs = fc('outputs', fc1_dropout, weights, 3)

    elif mode == 'inf':

        outputs = fc('outputs', fc1_relu, weights, 3)

    elif mode == 'tit':

        outputs = fc('outputs', fc1_relu, weights, 3, mode)

    confidence = mx.symbol.Activation(mx.symbol.slice_axis(outputs, axis = 1, begin = 0, end = 1), 'sigmoid')

    coordinate = mx.symbol.slice_axis(outputs, axis = 1, begin = 1, end = 3) # linear
    
    #return mx.symbol.concat(confidence, coordinate, dim = 1)

    return confidence, coordinate

def c3d(inputs, weights):

    conv1 = mx.symbol.Convolution(inputs, kernel = (3,3,3), stride = (1,1,1), pad = (1,1,1), num_filter = 64, name = 'conv1', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu1 = mx.symbol.Activation(conv1, act_type = 'relu')
    pool1 = mx.symbol.Pooling(relu1, pool_type = 'max', kernel = (1,2,2), stride = (1,2,2))
    
    ## 2nd group
    conv2 = mx.symbol.Convolution(pool1, kernel = (3,3,3), stride = (1,1,1), pad = (1,1,1), num_filter = 128, name = 'conv2', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu2 = mx.symbol.Activation(conv2, act_type = 'relu')
    pool2 = mx.symbol.Pooling(relu2, pool_type = 'max', kernel = (2,2,2), stride = (2,2,2))

    ## 3rd group
    conv3a = mx.symbol.Convolution(pool2, kernel = (3,3,3), stride = (1,1,1), pad = (1,1,1), num_filter = 256, name = 'conv3a', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu3a = mx.symbol.Activation(conv3a, act_type = 'relu')
    conv3b = mx.symbol.Convolution(relu3a, kernel = (3,3,3), stride = (1,1,1), pad = (1,0,0), num_filter = 256, name = 'conv3b', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu3b = mx.symbol.Activation(conv3b, act_type = 'relu')
    pool3b = mx.symbol.Pooling(relu3b, pool_type = 'max', kernel = (2,2,2), stride = (2,2,2))

    ## 4th group
    conv4a = mx.symbol.Convolution(pool3b, kernel = (3,3,3), stride = (1,1,1), pad = (1,1,1), num_filter = 512, name = 'conv4a', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu4a = mx.symbol.Activation(conv4a, act_type = 'relu')
    conv4b = mx.symbol.Convolution(relu4a, kernel = (3,3,3), stride = (1,1,1), pad = (1,0,0), num_filter = 512, name = 'conv4b', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu4b = mx.symbol.Activation(conv4b, act_type = 'relu')
    pool4b = mx.symbol.Pooling(relu4b, pool_type = 'max', kernel = (2,2,2), stride = (2,2,2))
    
    ## 5th group
    conv5a = mx.symbol.Convolution(pool4b, kernel = (3,3,3), stride = (1,1,1), pad = (1,0,0), num_filter = 512, name = 'conv5a', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu5a = mx.symbol.Activation(conv5a, act_type = 'relu')
    conv5b = mx.symbol.Convolution(relu5a, kernel = (3,3,3), stride = (1,1,1), pad = (1,0,0), num_filter = 512, name = 'conv5b', cudnn_tune = 'fastest', layout = 'NCDHW')
    relu5b = mx.symbol.Activation(conv5b, act_type = 'relu')
    pool5b = mx.symbol.Pooling(relu5b, pool_type = 'max', kernel = (2,2,2), stride = (2,2,2))
    
    ## 6th group

    fc6 = mx.symbol.FullyConnected(pool5b, num_hidden = 4096, name = 'fc6')
    relu6 = mx.symbol.Activation(fc6, act_type = 'relu')
    
    fc7 = mx.symbol.FullyConnected(relu6, num_hidden = 4096, name = 'fc7')
    relu7 = mx.symbol.Activation(data = fc7, act_type = 'relu')
    #drop7 = mx.symbol.Dropout(data = relu7, p = 0.5)

    #Loss
    #fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes)
    #softmax = mx.symbol.SoftmaxOutput(data=fc8, label=label, name='softmax')
    return relu7
