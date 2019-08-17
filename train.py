import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import shutil

FORMAT = 'NHWC'

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
    def __init__(self,in_dim,out_dim,out_activ):
        self.CONV_SIZE = [3,3]
        self.POOL_SHAPE = [2,2]
        self.conv1 = Conv2d(in_dim,out_dim,self.CONV_SIZE,default_activ)
        self.conv2 = Conv2d(out_dim,out_dim,self.CONV_SIZE,out_activ,strides=self.POOL_SHAPE)

    def calc(self,in_vec):
        cur_vec = in_vec
        cur_vec = self.conv1.calc(cur_vec)
        cur_vec = self.conv2.calc(cur_vec)
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
    vecs2 = tf.transpose(vecs2)
    dists = (tf.reduce_sum(sqr(vecs1), axis=1, keepdims=True)
             - 2 * tf.matmul(vecs1, vecs2)
             + tf.reduce_sum(sqr(vecs2), axis=0, keepdims=True))
    return dists

@tf.custom_gradient
def quant_calc(qu_vecs,in_vecs):
    dists = distances(in_vecs,qu_vecs)

    closest_vec_idx = tf.argmin(dists,axis=1)
    closest_vec_values = in_vecs#tf.gather(qu_vecs,closest_vec_idx,axis=0)

    def grad(dy):
        return tf.zeros_like(qu_vecs),dy
    return closest_vec_values,grad



class QuantBlock:
    def __init__(self,QUANT_SIZE,QUANT_DIM):
        init_vals = tf.random_normal([QUANT_SIZE,QUANT_DIM],dtype=tf.float32)
        self.vectors = tf.Variable(init_vals,name="vecs")
        self.vector_counts = tf.Variable(tf.zeros(shape=[QUANT_SIZE],dtype=tf.float32),name="vecs")
        self.QUANT_SIZE = QUANT_SIZE

    def calc(self, input):
        out_val = quant_calc(self.vectors,input)
        return out_val

    def calc_other_vals(self,input):
        distances = tf.matmul(input,self.vectors,transpose_b=True)

        closest_vec_idx = tf.argmin(distances,axis=1)
        closest_vec_values = tf.gather(self.vectors,closest_vec_idx)

        codebook_loss = tf.reduce_sum(sqr(closest_vec_values - tf.stop_gradient(input)))

        beta_val = 0.25 #from https://arxiv.org/pdf/1906.00446.pdf
        commitment_loss = tf.reduce_sum(beta_val * sqr(tf.stop_gradient(closest_vec_values) - input))

        idx_one_hot = tf.one_hot(closest_vec_idx,self.QUANT_SIZE)
        total = tf.reduce_sum(idx_one_hot,axis=0)
        update_counts = tf.assign(self.vector_counts,self.vector_counts+total)

        return commitment_loss + codebook_loss,update_counts

    #def resample_bad_vecs(self,input):
        #sample_vals = tf.random_normal([QUANT_SIZE,QUANT_DIM],dtype=tf.float32)
        #tf.equal(self.vector_counts)

def prod(l):
    p = 1
    for x in l:
        p *= x
    return p

class QuantBlockImg(QuantBlock):
    def calc(self,input):
        in_shape = input.get_shape().as_list()
        flat_val = tf.reshape(input,[prod(in_shape[:3]),in_shape[3]])
        out = QuantBlock.calc(self,flat_val)
        restored = tf.reshape(out,in_shape)
        return restored
    def calc_other_vals(self,input):
        in_shape = input.get_shape().as_list()
        flat_val = tf.reshape(input,[prod(in_shape[:3]),in_shape[3]])
        return QuantBlock.calc_other_vals(self,flat_val)


class MainCalc:
    def __init__(self):
        self.convpool1 = Convpool2(3,64,default_activ)
        self.convpool2 = Convpool2(64,64,None)
        self.quant_block = QuantBlockImg(128,64)
        self.convunpool1 = Deconv2(64,64,default_activ)
        self.convunpool2 = Deconv2(64,3,tf.nn.sigmoid)

    def calc(self,input):
        out1 = self.convpool1.calc(input)
        out2 = self.convpool2.calc(out1)
        quant = self.quant_block.calc(out2)
        print(quant.shape)
        dec1 = self.convunpool1.calc(out2)
        decoded_final = self.convunpool2.calc(dec1)

        reconstr_loss = tf.reduce_sum(sqr(decoded_final - input))

        quant_loss,update = self.quant_block.calc_other_vals(out2)
        tot_loss = reconstr_loss + quant_loss
        return update,tot_loss, reconstr_loss,decoded_final


mc = MainCalc()
place = tf.placeholder(shape=[4,384,512,3],dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

mc_update, loss, reconst_l, final_output = mc.calc(place)

opt = optimizer.minimize(loss)

orig_imgs = []
orig_filenames = []
for img_name in os.listdir("data/input_data"):
    with Image.open("data/input_data/"+img_name) as img:
        if img.size[1] == 384:
            orig_imgs.append(np.array(img).astype(np.float32)/256.0)
            orig_filenames.append(img_name)

fold_names = [fname.split('.')[0]+"/" for fname in orig_filenames]

for fold,fname in zip(fold_names,orig_filenames):
    fold_path = "data/result/"+fold
    os.makedirs(fold_path,exist_ok=True)
    shutil.copy("data/input_data/"+fname,fold_path+"org.jpg")

imgs = [img for img in orig_imgs]
saver = tf.train.Saver(max_to_keep=50)
SAVE_DIR = "data/save_model/"
os.makedirs(SAVE_DIR,exist_ok=True)
SAVE_NAME = SAVE_DIR+"model.ckpt"
#print(imgs[0])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    if os.path.exists(SAVE_DIR+"checkpoint"):
        print("reloaded")
        print(tf.train.latest_checkpoint(SAVE_DIR))
        saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

    batch = []
    print_num = 0
    while True:
        for x in range(2):
            random.shuffle(imgs)
            tot_loss = 0
            rec_loss = 0
            count = 0
            for img in imgs:
                batch.append(img)
                if len(batch) == 4:
                    count += 1
                    _,_,cur_loss,cur_rec = sess.run([mc_update,opt,loss,reconst_l],feed_dict={
                        place:np.stack(batch)
                    })
                    tot_loss += cur_loss
                    rec_loss += cur_rec
                    batch = []

            print("epoc ended, loss: {}   {}".format(tot_loss/count,rec_loss/count))

        print(",".join([str(val) for val in sess.run(mc.quant_block.vector_counts)[:20]]))
        print_num += 1
        saver.save(sess,SAVE_NAME,global_step=print_num)
        img_batch = []
        fold_batch = []
        for img,fold in zip(orig_imgs,fold_names):
            img_batch.append((img))
            fold_batch.append((fold))
            if len(img_batch) == 4:
                print("batch start")
                batch_outs = sess.run(final_output,feed_dict={
                    place:np.stack(img_batch)
                })
                pixel_vals = (batch_outs * 256).astype(np.uint8)
                for out,out_fold in zip(pixel_vals,fold_batch):
                    print(out.shape)
                    img = Image.fromarray(out)
                    img_path = "data/result/{}{}.jpg".format(out_fold,print_num)
                    print(img_path)
                    img.save(img_path)
                img_batch = []
                fold_batch = []

#print(out.shape)
