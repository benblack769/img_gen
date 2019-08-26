import tensorflow as tf

FORMAT = 'NCHW'

class Dense:
    def __init__(self,input_dim,out_dim,activation):
        out_shape = [input_dim,out_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name="weights")
        self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation

    def calc(self,input_vec):
        linval = tf.matmul(input_vec,self.weights) + self.biases
        return (linval if self.activation is None else
                    self.activation(linval))

    def vars(self):
        return [self.weights,self.biases]

class Conv2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        out_shape = conv_size+[input_dim]+[out_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name="weights")
        #self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation
        self.strides = strides
        self.padding = padding

    def calc(self,input_vec):
        linval = tf.nn.conv2d(
            input=input_vec,
            filter=self.weights,
            strides=self.strides,
            data_format=FORMAT,
            padding=self.padding)
        #affine_val = linval + self.biases
        activated = (linval if self.activation is None else
                    self.activation(linval))
        return activated

    def vars(self):
        return [self.weights,self.biases]


def Conv1x1(input_dim,out_dim,activation):
    return Conv2d(input_dim,out_dim,[1,1],activation)

def Conv1x1Upsample(input_dim,out_dim,activation,out_shape,upsample_factor):
    return ConvTrans2d(input_dim,out_dim,[1,1],activation,out_shape,strides=[upsample_factor,upsample_factor])


class ConvTrans2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,out_shape,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        filter_shape = conv_size+[out_dim]+[input_dim]
        init_vals = tf.initializers.glorot_normal()(filter_shape)
        self.weights = tf.Variable(init_vals,name="weights")
        #self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.out_dim = out_dim
        self.out_shape = out_shape


    def calc(self,input_vec):
        in_shape = input_vec.get_shape().as_list()
        out_shape = [
            in_shape[0],
            self.out_dim,
            self.out_shape[0],
            self.out_shape[1],
        ]
        linval = tf.nn.conv2d_transpose(
            value=input_vec,
            filter=self.weights,
            output_shape=out_shape,
            strides=self.strides,
            data_format=FORMAT)
        #affine_val = linval + self.biases
        activated = (linval if self.activation is None else
                    self.activation(linval))
        return activated

    def vars(self):
        return [self.weights,self.biases]

def avgpool2d(input,window_shape):
    return tf.nn.pool(input,
        window_shape=window_shape,
        pooling_type="AVG",
        padding="SAME",
        strides=window_shape,
        )

def unpool(tens4d,factor):
    shape = tens4d.get_shape().as_list()
    spread_shape = [shape[0],shape[1],1,shape[2],1,shape[3]]
    reshaped = tf.reshape(tens4d,spread_shape)
    tiled = tf.tile(reshaped,[1,1,factor,1,factor,1])
    new_shape = [shape[0],shape[1],factor*shape[2],factor*shape[3]]
    back = tf.reshape(tiled,new_shape)
    return back

def default_activ(input):
    return tf.nn.relu(input)


class Convpool2:
    def __init__(self,in_dim,out_dim,out_activ,use_batchnorm=True):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.out_activ = out_activ
        self.use_batchnorm = use_batchnorm
        #self.bn1 = tf.layers.BatchNormalization(momentum=0.9)
        #if self.use_batchnorm:
        #    self.bn2 = tf.layers.BatchNormalization(momentum=0.9)
        self.conv1 = Conv2d(in_dim,out_dim,self.CONV_SIZE,None)
        self.conv2 = Conv2d(out_dim,out_dim,self.CONV_SIZE,None,strides=self.POOL_SHAPE)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        #cur_vec = self.bn1(cur_vec)
        cur_vec = default_activ(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
        #if self.use_batchnorm:
        #    cur_vec = self.bn2(cur_vec)
        if self.out_activ is not None:
            cur_vec = self.out_activ(cur_vec)
        #cur_vec = avgpool2d(cur_vec,self.POOL_SHAPE)
        return cur_vec


class Deconv2:
    def __init__(self,in_dim,out_dim,out_activ,out_shape):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.conv1 = ConvTrans2d(in_dim,in_dim,self.CONV_SIZE,default_activ,out_shape,strides=self.POOL_SHAPE)
        self.conv2 = ConvTrans2d(in_dim,out_dim,self.CONV_SIZE,out_activ,out_shape)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
        #cur_vec = avgpool2d(cur_vec,self.POOL_SHAPE)
        return cur_vec
