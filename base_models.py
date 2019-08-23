import tensorflow as tf

FORMAT = 'NHWC'
def default_activ(x):
    return tf.nn.relu(x)

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
    def __init__(self,input_dim,out_dim,conv_size,activation,strides=[1,1],padding="SAME",name=None):
        name = name if name is not None else "weights"
        assert len(conv_size) == 2,"incorrect conv size"
        out_shape = conv_size+[input_dim]+[out_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name=name)
        self.bn = tf.layers.BatchNormalization()
        self.activation = activation
        self.strides = strides
        self.padding = padding

    def calc(self,input_vec):
        linval = tf.nn.conv2d(
            input=input_vec,
            filter=self.weights,
            strides=[1]+self.strides+[1],
            data_format=FORMAT,
            padding=self.padding)
        normalized = self.bn(linval)
        activated = (normalized if self.activation is None else
                    self.activation(normalized))
        return activated

    def vars(self):
        return [self.weights]

def Conv1x1(input_dim,out_dim,activation):
    return Conv2d(input_dim,out_dim,[1,1],activation)

class ConvTrans2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,out_shape,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        filter_shape = conv_size+[out_dim]+[input_dim]
        init_vals = tf.initializers.glorot_normal()(filter_shape)
        self.weights = tf.Variable(init_vals,name="weights")
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.out_dim = out_dim
        self.out_shape = out_shape

    def calc(self,input_vec):
        in_shape = input_vec.get_shape().as_list()
        out_shape = [
            in_shape[0],
            self.out_shape[0],
            self.out_shape[1],
            self.out_dim
        ]
        linval = tf.nn.conv2d_transpose(
            value=input_vec,
            filter=self.weights,
            output_shape=out_shape,
            strides=[1]+self.strides+[1],
            data_format=FORMAT)
        #affine_val = linval + self.biases
        activated = (linval if self.activation is None else
                    self.activation(linval))
        return activated

    def vars(self):
        return [self.weights]

def avgpool2d(input,window_shape):
    return tf.nn.pool(input,
        window_shape=window_shape,
        pooling_type="AVG",
        padding="SAME",
        strides=window_shape,
        )

class Convpool2:
    def __init__(self,in_dim,out_dim,out_activ,use_batchnorm=True,name=""):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.out_activ = out_activ
        self.use_batchnorm = use_batchnorm
        self.conv1 = Conv2d(in_dim,out_dim,self.CONV_SIZE,None,name=name+"1")
        self.conv2 = Conv2d(out_dim,out_dim,self.CONV_SIZE,None,name=name+"2",strides=self.POOL_SHAPE)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = default_activ(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
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
