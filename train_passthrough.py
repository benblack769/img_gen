import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import shutil

FORMAT = 'NHWC'
BATCH_SIZE = 8

def sqr(x):
    return x * x

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
            strides=[1]+self.strides+[1],
            data_format=FORMAT,
            padding=self.padding)
        #affine_val = linval + self.biases
        activated = (linval if self.activation is None else
                    self.activation(linval))
        return activated

    def vars(self):
        return [self.weights,self.biases]


class ConvTrans2d:
    def __init__(self,input_dim,out_dim,conv_size,activation,strides=[1,1],padding="SAME"):
        assert len(conv_size) == 2,"incorrect conv size"
        out_shape = conv_size+[out_dim]+[input_dim]
        init_vals = tf.initializers.glorot_normal()(out_shape)
        self.weights = tf.Variable(init_vals,name="weights")
        #self.biases = tf.Variable(tf.ones(out_dim)*0.01,name="biases")
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.out_dim = out_dim

    def calc(self,input_vec):
        out_shape = input_vec.get_shape().as_list()
        print(out_shape)
        out_shape[1] *= self.strides[0]
        out_shape[2] *= self.strides[1]
        out_shape[3] = self.out_dim
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
        return [self.weights,self.biases]

def avgpool2d(input,window_shape):
    return tf.nn.pool(input,
        window_shape=window_shape,
        pooling_type="AVG",
        padding="SAME",
        strides=window_shape,
        )

def default_activ(input):
    return tf.nn.relu(input)


class Convpool2:
    def __init__(self,in_dim,out_dim,out_activ,use_batchnorm=True):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.out_activ = out_activ
        self.use_batchnorm = use_batchnorm
        self.bn1 = tf.layers.BatchNormalization()
        if self.use_batchnorm:
            self.bn2 = tf.layers.BatchNormalization()
        self.conv1 = Conv2d(in_dim,out_dim,self.CONV_SIZE,None)
        self.conv2 = Conv2d(out_dim,out_dim,self.CONV_SIZE,None,strides=self.POOL_SHAPE)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = self.bn1(cur_vec)
        cur_vec = default_activ(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
        if self.use_batchnorm:
            cur_vec = self.bn2(cur_vec)*0.1
        if self.out_activ is not None:
            cur_vec = self.out_activ(cur_vec)
        #cur_vec = avgpool2d(cur_vec,self.POOL_SHAPE)
        return cur_vec


class Deconv2:
    def __init__(self,in_dim,out_dim,out_activ):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.conv1 = ConvTrans2d(in_dim,in_dim,self.CONV_SIZE,default_activ,strides=self.POOL_SHAPE)
        self.conv2 = ConvTrans2d(in_dim,out_dim,self.CONV_SIZE,out_activ)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
        #cur_vec = avgpool2d(cur_vec,self.POOL_SHAPE)
        return cur_vec


def distances(vecs1,vecs2):
    return tf.matmul(vecs1,vecs2,transpose_b=True)
    vecs2 = tf.transpose(vecs2)
    dists = (tf.reduce_sum(sqr(vecs1), axis=1, keepdims=True)
             - 2 * tf.matmul(vecs1, vecs2)
             + tf.reduce_sum(sqr(vecs2), axis=0, keepdims=True))
    return dists

def gather_multi_idxs(qu_vecs,chosen_idxs):
    idx_shape = chosen_idxs.get_shape().as_list()
    qu_shape = qu_vecs.get_shape().as_list()
    idx_add = tf.range(qu_shape[0],dtype=tf.int64)*qu_shape[1] + chosen_idxs
    idx_transform = tf.reshape(idx_add,[prod(idx_shape)])
    rqu_vecs = tf.reshape(qu_vecs,[qu_shape[0]*qu_shape[1],qu_shape[2]])

    closest_vec_values = tf.gather(rqu_vecs,idx_transform,axis=0)

    combined_vec_vals = tf.reshape(closest_vec_values,[idx_shape[0],qu_shape[0]*qu_shape[2]])

    return combined_vec_vals

@tf.custom_gradient
def quant_calc(qu_vecs,chosen_idxs,in_vecs):
    closest_vec_values = gather_multi_idxs(qu_vecs,chosen_idxs)

    def grad(dy):
        return tf.zeros_like(qu_vecs),tf.zeros_like(chosen_idxs),dy
    return closest_vec_values,grad

class QuantBlock:
    def __init__(self,QUANT_SIZE,NUM_QUANT,QUANT_DIM):
        init_vals = tf.random_normal([NUM_QUANT,QUANT_SIZE,QUANT_DIM],dtype=tf.float32)*0.1
        self.vectors = tf.Variable(init_vals,name="vecs")
        self.vector_counts = tf.Variable(tf.zeros(shape=[NUM_QUANT,QUANT_SIZE],dtype=tf.float32),name="vecs")
        self.QUANT_SIZE = QUANT_SIZE
        self.QUANT_DIM = QUANT_DIM
        self.NUM_QUANT = NUM_QUANT

    def calc(self, input):
        orig_size = input.get_shape().as_list()
        div_input = tf.reshape(input,[orig_size[0],self.NUM_QUANT,self.QUANT_DIM])
        dists = tf.einsum("ijk,jmk->ijm",div_input,self.vectors)

        #dists = distances(input,self.vectors)

        #soft_vals = tf.softmax(,axis=1)
        #inv_dists = 1.0/(dists+0.000001)
        #closest_vec_idx = tf.multinomial((inv_dists),1)
        #closest_vec_idx = tf.reshape(closest_vec_idx,shape=[closest_vec_idx.get_shape().as_list()[0]])
        #print(closest_vec_idx.shape)
        closest_vec_idx = tf.argmin(dists,axis=-1)

        out_val = quant_calc(self.vectors,closest_vec_idx,input)
        other_losses, update = self.calc_other_vals(input,closest_vec_idx)
        return out_val, other_losses, update

    def calc_other_vals(self,input,closest_vec_idx):
        closest_vec_values = gather_multi_idxs(self.vectors,closest_vec_idx)

        codebook_loss = tf.reduce_sum(sqr(closest_vec_values - tf.stop_gradient(input)))

        beta_val = 0.25 #from https://arxiv.org/pdf/1906.00446.pdf
        commitment_loss = tf.reduce_sum(beta_val * sqr(tf.stop_gradient(closest_vec_values) - input))

        idx_one_hot = tf.one_hot(closest_vec_idx,self.QUANT_SIZE)
        total = tf.reduce_sum(idx_one_hot,axis=0)
        update_counts = tf.assign(self.vector_counts,self.vector_counts+total)

        return commitment_loss + codebook_loss,update_counts

    def resample_bad_vecs(self):
        #sample_vals = tf.random_normal([self.QUANT_SIZE,self.QUANT_DIM],dtype=tf.float32)
        #equal_vals = tf.cast(tf.equal(self.vector_counts,0),dtype=tf.float32)
        #equal_vals= tf.reshape(equal_vals,shape=[self.QUANT_SIZE,1])
        #new_vecs = self.vectors - self.vectors * equal_vals + sample_vals * equal_vals
        #vec_assign = tf.assign(self.vectors,new_vecs)
        zero_assign = tf.assign(self.vector_counts,tf.zeros_like(self.vector_counts))
        #tot_assign = tf.group([vec_assign,zero_assign])
        return zero_assign#tot_assign

def prod(l):
    p = 1
    for x in l:
        p *= x
    return p

class QuantBlockImg(QuantBlock):
    def calc(self,input):
        in_shape = input.get_shape().as_list()
        flat_val = tf.reshape(input,[prod(in_shape[:3]),in_shape[3]])
        out,o1,o2 = QuantBlock.calc(self,flat_val)
        restored = tf.reshape(out,in_shape)
        return restored,o1,o2

class MainCalc:
    def __init__(self):
        self.convpool1 = Convpool2(3,64,default_activ)
        self.convpool2 = Convpool2(64,128,None)
        self.quant_block = QuantBlockImg(128,4,32)
        self.convunpool1 = Deconv2(128,64,default_activ)
        self.convunpool2 = Deconv2(64,3,tf.nn.sigmoid)

    def calc(self,input):
        out1 = self.convpool1.calc(input)
        out2 = self.convpool2.calc(out1)
        quant,quant_loss,update = self.quant_block.calc(out2)
        print(quant.shape)
        dec1 = self.convunpool1.calc(out2)
        decoded_final = self.convunpool2.calc(dec1)

        reconstr_loss = tf.reduce_sum(sqr(decoded_final - input))

        tot_loss = reconstr_loss + quant_loss
        return update,tot_loss, reconstr_loss,decoded_final

    def periodic_update(self):
        return self.quant_block.resample_bad_vecs()

mc = MainCalc()
place = tf.placeholder(shape=[BATCH_SIZE,200,320,3],dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

mc_update, loss, reconst_l, final_output = mc.calc(place)
resample_update = mc.periodic_update()

opt = optimizer.minimize(loss)
orig_imgs = []
orig_filenames = []
for img_name in os.listdir("data/input_data"):
    with Image.open("data/input_data/"+img_name) as img:
        if img.mode == "RGB":
            orig_imgs.append(np.array(img).astype(np.float32)/256.0)
            orig_filenames.append(img_name)

fold_names = [fname.split('.')[0]+"/" for fname in orig_filenames[:50]]

for fold,fname in zip(fold_names,orig_filenames):
    fold_path = "data/result/"+fold
    os.makedirs(fold_path,exist_ok=True)
    shutil.copy("data/input_data/"+fname,fold_path+"org.jpg")

imgs = [img for img in orig_imgs]
saver = tf.train.Saver(max_to_keep=50)
SAVE_DIR = "data/save_model/"
os.makedirs(SAVE_DIR,exist_ok=True)
SAVE_NAME = SAVE_DIR+"model.ckpt"
logfilename = "data/count_log.txt"
logfile = open(logfilename,'w')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print_num = 0
    if os.path.exists(SAVE_DIR+"checkpoint"):
        print("reloaded")
        checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
        print(checkpoint)
        print_num = int(checkpoint.split('-')[1])
        saver.restore(sess, checkpoint)

    batch = []
    batch_count = 0
    while True:
        for x in range(20):
            random.shuffle(imgs)
            tot_loss = 0
            rec_loss = 0
            loss_count = 0
            for img in imgs:
                batch.append(img)
                if len(batch) == BATCH_SIZE:
                    batch_count += 1
                    _,_,cur_loss,cur_rec = sess.run([mc_update,opt,loss,reconst_l],feed_dict={
                        place:np.stack(batch)
                    })
                    loss_count += 1
                    tot_loss += cur_loss
                    rec_loss += cur_rec
                    batch = []

                    EPOC_SIZE = 100
                    if batch_count % EPOC_SIZE == 0:
                        print("epoc ended, loss: {}   {}".format(tot_loss/loss_count,rec_loss/loss_count))
                        logfile.write(",".join([str(val) for val in sess.run(mc.quant_block.vector_counts)]))
                        logfile.flush()
                        sess.run(resample_update)

                        tot_loss = 0
                        rec_loss = 0
                        loss_count = 0

                        if batch_count % (EPOC_SIZE*10) == 0:
                            print_num += 1
                            print("save {} started".format(print_num))
                            saver.save(sess,SAVE_NAME,global_step=print_num)
                            img_batch = []
                            fold_batch = []
                            for count,(img,fold) in enumerate(zip(orig_imgs,fold_names)):
                                img_batch.append((img))
                                fold_batch.append((fold))
                                if len(img_batch) == BATCH_SIZE:
                                    batch_outs = sess.run(final_output,feed_dict={
                                        place:np.stack(img_batch)
                                    })
                                    pixel_vals = (batch_outs * 256).astype(np.uint8)
                                    for out,out_fold in zip(pixel_vals,fold_batch):
                                        #print(out.shape)
                                        img = Image.fromarray(out)
                                        img_path = "data/result/{}{}.jpg".format(out_fold,print_num)
                                        #print(img_path)
                                        img.save(img_path)
                                    img_batch = []
                                    fold_batch = []
                            print("save {} finished".format(print_num))


#print(out.shape)
