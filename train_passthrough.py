import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import shutil
from base_models import Dense,Conv2d,Conv1x1,ConvTrans2d,Convpool2,Deconv2,default_activ

BATCH_SIZE = 8
IMG_SIZE = (200,320)

def sqr(x):
    return x * x

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
        init_vals = tf.random.normal(shape=[NUM_QUANT,QUANT_SIZE,QUANT_DIM],dtype=tf.float32)*0.1
        self.vectors = tf.Variable(init_vals,name="vecs")
        self.QUANT_SIZE = QUANT_SIZE
        self.QUANT_DIM = QUANT_DIM
        self.NUM_QUANT = NUM_QUANT

    def calc(self, input):
        input = input*0.1
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
        other_losses = self.calc_other_vals(input,closest_vec_idx)
        return out_val, other_losses,closest_vec_idx

    def calc_other_vals(self,input,closest_vec_idx):
        closest_vec_values = gather_multi_idxs(self.vectors,closest_vec_idx)

        codebook_loss = tf.reduce_sum(sqr(closest_vec_values - tf.stop_gradient(input)))

        beta_val = 0.25 #from https://arxiv.org/pdf/1906.00446.pdf
        commitment_loss = tf.reduce_sum(beta_val * sqr(tf.stop_gradient(closest_vec_values) - input))

        return commitment_loss + codebook_loss

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

IMG_LEVEL = 32
SECOND_LEVEL = 64
THIRD_LEVEL = 128
FOURTH_LEVEL = 192
FIFTH_LEVEL = 256
ZIXTH_LEVEL = 256

def round_up_div(num,denom):
    return (num+denom-1) // denom

def get_out_dim(dim,level):
    return round_up_div(dim, 2**(level-1))

def get_out_shape(level):
    return (get_out_dim(IMG_SIZE[0],level),get_out_dim(IMG_SIZE[1],level))

class MainCalc:
    def __init__(self):
        with tf.variable_scope("save_layers"):
            self.convpool1 = Convpool2(3,IMG_LEVEL,default_activ,name="1")
            self.convpool2 = Convpool2(IMG_LEVEL,SECOND_LEVEL,None,name="2")
            self.convpool3 = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ,name="3")
            self.convpool4 = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,None,name="4")
            self.convpool5 = Convpool2(FOURTH_LEVEL,FIFTH_LEVEL,default_activ,name="5")
            self.convpool6 = Convpool2(FIFTH_LEVEL,ZIXTH_LEVEL,None,name="6")

        self.quanttrans1 = Conv1x1(SECOND_LEVEL,SECOND_LEVEL,None)
        self.deconvbn1 = tf.layers.BatchNormalization()
        self.quant_block1 = QuantBlockImg(256,1,SECOND_LEVEL)
        self.quanttrans2 = Conv1x1(FOURTH_LEVEL,FOURTH_LEVEL,None)
        self.deconvbn2 = tf.layers.BatchNormalization()
        self.quant_block2 = QuantBlockImg(256,4,FOURTH_LEVEL//4)
        self.quant_block3 = QuantBlockImg(256,4,ZIXTH_LEVEL//4)

        self.deconv6 = Deconv2(ZIXTH_LEVEL,FIFTH_LEVEL,default_activ,get_out_shape(6))
        self.deconv5 = Deconv2(FIFTH_LEVEL,FOURTH_LEVEL,default_activ,get_out_shape(5))
        self.deconv4 = Deconv2(FOURTH_LEVEL,THIRD_LEVEL,default_activ,get_out_shape(4))
        self.deconv3 = Deconv2(THIRD_LEVEL,SECOND_LEVEL,default_activ,get_out_shape(3))
        self.deconv2 = Deconv2(SECOND_LEVEL,IMG_LEVEL,default_activ,get_out_shape(2))
        self.deconv1 = Deconv2(IMG_LEVEL,3,tf.sigmoid,get_out_shape(1))

    def calc(self,input):
        out1 = self.convpool1.calc(input)
        out2 = self.convpool2.calc(out1)
        '''out3 = self.convpool3.calc(out2)
        out4 = self.convpool4.calc(out3)
        out5 = self.convpool5.calc(out4)
        out6 = self.convpool6.calc(out5)

        quant3,quant_loss3,closest3 = self.quant_block3.calc(out6)
        dec6 = self.deconv6.calc(quant3)
        dec5 = self.deconv5.calc(dec6)
        out4trans = self.quanttrans2.calc(out4)
        quant2,quant_loss2,closest2 = self.quant_block2.calc(dec5+out4trans)
        dec4 = self.deconv4.calc(quant2)
        dec3 = self.deconv3.calc(dec4)
        out2trans = self.quanttrans1.calc(out2)'''
        quant1,quant_loss1,closest1 = self.quant_block1.calc(out2)#dec3+out2trans)
        dec2 = self.deconv2.calc(quant1)
        dec1 = self.deconv1.calc(dec2)
        decoded_final = dec1

        reconstr_loss = tf.reduce_sum(sqr(decoded_final - input))

        quant_loss = quant_loss1 #+ quant_loss2 + quant_loss3
        tot_loss = reconstr_loss + quant_loss
        closest_list = [closest1]#,closest2,closest3]
        return tot_loss, reconstr_loss,decoded_final,closest_list


mc = MainCalc()
place = tf.placeholder(shape=[BATCH_SIZE,200,320,3],dtype=tf.float32)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

loss, reconst_l, final_output, closest_list = mc.calc(place)

opt = optimizer.minimize(loss)
orig_imgs = []
orig_filenames = []
for img_name in os.listdir("data/input_data")[:1000]:
    with Image.open("data/input_data/"+img_name) as img:
        if img.mode == "RGB":
            orig_imgs.append(np.array(img).astype(np.float32)/256.0)
            orig_filenames.append(img_name)

full_fold_names = [fname.split('.')[0]+"/" for fname in orig_filenames]
fold_names = full_fold_names[:50]

for fold,fname in zip(fold_names,orig_filenames):
    fold_path = "data/result/"+fold
    os.makedirs(fold_path,exist_ok=True)
    shutil.copy("data/input_data/"+fname,fold_path+"org.jpg")

TRAIN = True
if TRAIN:
    imgs = [img for img in orig_imgs]
    full_saver = tf.train.Saver(max_to_keep=20)
    SAVE_DIR = "data/save_model/"
    os.makedirs(SAVE_DIR,exist_ok=True)
    SAVE_NAME = SAVE_DIR+"model.ckpt"

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
            full_saver.restore(sess, checkpoint)

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
                        _,cur_loss,cur_rec = sess.run([opt,loss,reconst_l],feed_dict={
                            place:np.stack(batch)
                        })
                        loss_count += 1
                        tot_loss += cur_loss
                        rec_loss += cur_rec
                        batch = []

                        EPOC_SIZE = 100
                        if batch_count % EPOC_SIZE == 0:
                            print("epoc ended, loss: {}   {}".format(tot_loss/loss_count,rec_loss/loss_count),flush=True)

                            tot_loss = 0
                            rec_loss = 0
                            loss_count = 0

                            if batch_count % (EPOC_SIZE*10) == 0:
                                print_num += 1
                                print("save {} started".format(print_num))
                                full_saver.save(sess,SAVE_NAME,global_step=print_num)
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
else:
    out_path = "data/pretrained_result/"
    os.mkdir(out_path)
    full_saver = tf.train.Saver(max_to_keep=20)
    SAVE_DIR = "data/save_model/"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
        print(checkpoint)
        print_num = int(checkpoint.split('-')[1])
        saver.restore(sess, checkpoint)
        for idx in range(0,len(imgs),BATCH_SIZE):
            batch = tf.stack(imgs[idx:idx+BATCH_SIZE])

            out_closest_list = sess.run(*closest_list,feed_dict={
                place:np.stack(img_batch)
            })
            for bidx in range(BATCH_SIZE):
                img_idx = bidx + idx
                new_path = out_path+full_fold_names[img_idx]
                os.mkdir(new_path)
                image_data = (imgs[img_idx]*256).astype(np.uint8)
                closest1,closest2,closest3 = [close.astype(np.uint16) for close in out_closest_list]
                np.save(new_path+"image.npy",image_data)
                np.save(new_path+"closest1.npy",closest1)
                np.save(new_path+"closest2.npy",closest2)
                np.save(new_path+"closest3.npy",closest3)
