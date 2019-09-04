import tensorflow as tf

FORMAT = 'NHWC'

class Dense:
    def __init__(self,input_dim,out_dim,activation):
        out_shape = [input_dim,out_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name="weights",use_resource=True)
        self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases",use_resource=True)
        self.activation = activation

    def calc(self,input_vec):
        linval = tf.matmul(input_vec,self.weights) + self.biases
        return (linval if self.activation is None else
                    self.activation(linval))

    def vars(self):
        return [self.weights,self.biases]

def magnitude(vec):
    assert len(vec.get_shape().as_list()) == 1
    return tf.tensordot(vec,vec,axes=1)

def conv_in_out(inp,weights):
    out = tf.nn.conv2d(input=inp,filter=weights,padding="VALID")
    take_dim = out.get_shape().as_list()[1]//2
    magn = magnitude(out[0][take_dim][take_dim])
    out = out / magn
    return out,magn

def conv_out_in(out,weights):
    filt_shape = weights.get_shape().as_list()
    filt_size = filt_shape[0]
    in_size = filt_shape[2]
    take_dim = out.get_shape().as_list()[1]//2
    out_shape = [1,out.shape[1]+filt_size-1,out.shape[2]+filt_size-1,in_size]
    inp = tf.nn.conv2d_transpose(value=out,filter=weights,output_shape=out_shape,padding="VALID")
    magn = magnitude(out[0][take_dim][take_dim])
    inp = inp / magn
    return inp,magn

class Conv2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        out_shape = conv_size+[input_dim]+[out_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name="weights",use_resource=True)
        self.biases = tf.Variable(tf.ones([out_dim])*0.01,name="biases",use_resource=True)
        self.specnorm_u = tf.Variable(tf.ones([input_dim])/input_dim,name="specnorm_u",use_resource=True)
        self.decay = tf.Variable(tf.ones([]),name="decay",use_resource=True)
        self.bn = tf.layers.BatchNormalization(axis=1)
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.input_dim = input_dim
        self.out_dim = out_dim

    def updates(self):
        cur_u = tf.reshape(self.specnorm_u,[1,1,1,self.input_dim])
        INP_SIZE = 7
        cur_u = tf.tile(cur_u,[1,INP_SIZE,INP_SIZE,1])
        print("shape")
        print(cur_u.shape)
        cur_v,_ = conv_in_out(cur_u,self.weights)
        print(cur_v.shape)
        cur_u,_ = conv_out_in(cur_v,self.weights)
        print(cur_u.shape)
        cur_v,_ = conv_in_out(cur_u,self.weights)
        print(cur_v.shape)
        cur_u,mag = conv_out_in(cur_v,self.weights)
        print(cur_u.shape)

        DECAY = self.decay
        new_weights = self.weights*DECAY + (1-DECAY)*self.weights/tf.maximum(1.0,mag)
        update_weights = tf.assign(self.weights,new_weights)
        update_u_vec = tf.assign(self.specnorm_u,cur_u[0][0][0])
        STEPS_TO_DECAY_HALF = 100
        decay_decay = 0.5**(1.0/STEPS_TO_DECAY_HALF)
        decay_update = tf.assign(self.decay,self.decay*decay_decay)
        return [update_weights,update_u_vec]

    def calc(self,input_vec):
        #self.updates()
        linval = tf.nn.conv2d(
            input=input_vec,
            filter=self.weights,
            strides=self.strides,
            data_format=FORMAT,
            padding=self.padding)
        linval = self.bn(linval)
        linval = linval + self.biases
        activated = (linval if self.activation is None else
                    self.activation(linval))
        return activated

    def vars(self,name):
        return [
            (name+"filters",self.weights),
            #(name+"biases",self.biases),
            #(name+"specnorm_u",self.specnorm_u)
        ]


def Conv1x1(input_dim,out_dim,activation):
    return Conv2d(input_dim,out_dim,[1,1],activation)

def Conv1x1Upsample(input_dim,out_dim,activation,out_shape,upsample_factor):
    return ConvTrans2d(input_dim,out_dim,[1,1],activation,out_shape,strides=[upsample_factor,upsample_factor])


class ConvTrans2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,out_shape,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        filter_shape = conv_size+[out_dim]+[input_dim]
        init_vals = tf.initializers.glorot_normal()(filter_shape)
        self.weights = tf.Variable(init_vals,name="weights",use_resource=True)
        self.specnorm_u = tf.Variable(tf.ones([input_dim])/input_dim,name="specnorm_u",use_resource=True)
        #self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.out_dim = out_dim
        self.out_shape = out_shape
        self.input_dim = input_dim

    def updates(self):
        return []
        ITERS = 3
        cur_u = tf.reshape(self.specnorm_u,[1,1,1,self.input_dim])
        INP_SIZE = 7
        cur_v = tf.tile(cur_u,[1,INP_SIZE,INP_SIZE,1])
        print("shape")
        cur_u,_ = conv_out_in(cur_v,self.weights)
        print(cur_u.shape)
        cur_v,_ = conv_in_out(cur_u,self.weights)
        print(cur_v.shape)
        cur_u,_ = conv_out_in(cur_v,self.weights)
        print(cur_u.shape)
        cur_v,mag = conv_in_out(cur_u,self.weights)
        print(cur_v.shape)

        #DECAY = 0.5
        new_weights = self.weights/tf.maximum(1.0,mag+0.001)
        update_weights = tf.assign(self.weights,new_weights)
        update_u_vec = tf.assign(self.specnorm_u,cur_v[0][0][0])
        return [update_weights,update_u_vec]

    def calc(self,input_vec):
        in_shape = input_vec.get_shape().as_list()
        out_shape = [
            in_shape[0],
            self.out_shape[0],
            self.out_shape[1],
            self.out_dim,
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

    def vars(self,name):
        return [(name,self.weights)]


def avgpool2d(input,window_shape):
    return tf.nn.pool(input,
        window_shape=window_shape,
        pooling_type="AVG",
        padding="SAME",
        strides=window_shape,
        )

def default_activ(input):
    return tf.nn.leaky_relu(input)


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

    def updates(self):
        return (
            self.conv1.updates() +
            self.conv2.updates()
        )

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

    def vars(self,name):
        return self.conv1.vars(name+"l1")+self.conv2.vars(name+"l2")


class Deconv2:
    def __init__(self,in_dim,out_dim,out_activ,out_shape):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.conv1 = ConvTrans2d(in_dim,in_dim,self.CONV_SIZE,default_activ,out_shape,strides=self.POOL_SHAPE)
        self.conv2 = ConvTrans2d(in_dim,out_dim,self.CONV_SIZE,out_activ,out_shape)

    def updates(self):
        return (
            self.conv1.updates() +
            self.conv2.updates()
        )

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
        #cur_vec = avgpool2d(cur_vec,self.POOL_SHAPE)
        return cur_vec

    def vars(self,name):
        return self.conv1.vars(name+"l1")+self.conv2.vars(name+"l2")
