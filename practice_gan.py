import os
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import shutil
from base_ops import default_activ,Convpool2,Conv2d,Conv1x1,Conv1x1Upsample,ConvTrans2d,Convpool2,Deconv2
from quant_block import QuantBlockImg
from npy_saver import NpySaver

BATCH_SIZE = 64

IMG_SIZE = (96,96)

def round_up_div(num,denom):
    return (num+denom-1) // denom

def get_out_dim(dim,level):
    return round_up_div(dim, 2**(level-1))

def get_out_shape(level):
    return [get_out_dim(IMG_SIZE[0],level),get_out_dim(IMG_SIZE[1],level)]

def sqr(x):
    return x * x

IMG_LEVEL = 32
SECOND_LEVEL = 64
THIRD_LEVEL = 128
FOURTH_LEVEL = 192
FIFTH_LEVEL = 256
ZIXTH_LEVEL = 256

class Discrim:
    def __init__(self):
        self.convpool1img = Convpool2(3,IMG_LEVEL,default_activ)
        self.convpool2img = Convpool2(IMG_LEVEL,SECOND_LEVEL,default_activ)
        self.convpool3img = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ)
        self.convpool4img = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,default_activ)
        self.comb_assess = Conv1x1(FOURTH_LEVEL,1,None)

    def calc(self,img):
        cur_img_out = img
        cur_img_out = self.convpool1img.calc(cur_img_out)
        cur_img_out = self.convpool2img.calc(cur_img_out)
        cur_img_out = self.convpool3img.calc(cur_img_out)
        cur_img_out = self.convpool4img.calc(cur_img_out)
        fin_assess = self.comb_assess.calc(cur_img_out)

        return fin_assess

    def updates(self):
        return (
            self.convpool1img.updates() +
            self.convpool2img.updates() +
            self.convpool3img.updates() +
            self.convpool4img.updates() #+
            #self.comb_assess.updates()
        )

    def vars(self):
        var_names = (
            self.convpool1img.vars("") +
            self.convpool2img.vars("") +
            self.convpool3img.vars("") +
            self.convpool4img.vars("") +
            self.comb_assess.vars("")
        )
        vars = [var for name, var in var_names]
        return vars

RAND_SIZE = 16
class Gen:
    def __init__(self):
        self.deconv4 = Deconv2(RAND_SIZE,THIRD_LEVEL,default_activ,get_out_shape(4))
        self.deconv3 = Deconv2(THIRD_LEVEL,SECOND_LEVEL,default_activ,get_out_shape(3))
        self.deconv2 = Deconv2(SECOND_LEVEL,IMG_LEVEL,default_activ,get_out_shape(2))
        self.deconv1 = Deconv2(IMG_LEVEL,3,tf.sigmoid,get_out_shape(1))

    def calc(self):

        rand_inp = tf.random.normal(shape=[BATCH_SIZE]+get_out_shape(5)+[RAND_SIZE])

        deconv4 = self.deconv4.calc(rand_inp)
        deconv3 = self.deconv3.calc(deconv4)
        deconv2 = self.deconv2.calc(deconv3)
        deconv1 = self.deconv1.calc(deconv2)

        fin_out = deconv1

        return fin_out

    def updates(self):
        return (
            self.deconv4.updates() +
            self.deconv3.updates() +
            self.deconv2.updates() +
            self.deconv1.updates()
        )

    def vars(self):
        var_names = (
            self.deconv4.vars("") +
            self.deconv3.vars("") +
            self.deconv2.vars("") +
            self.deconv1.vars("")
        )
        vars = [var for name, var in var_names]
        return vars

class MainCalc:
    def __init__(self):
        self.gen = Gen()
        self.discrim = Discrim()
        self.discrim_optim = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9)
        self.gen_optim = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9)
        self.bn_grads = tf.layers.BatchNormalization(axis=1)

    def updates(self):
        return  self.discrim.updates()#+self.gen.updates()

    def calc_loss(self,true_imgs):
        new_img = self.gen.calc()

        true_diffs = self.discrim.calc(true_imgs)
        false_diffs = self.discrim.calc(new_img)

        all_diffs = tf.concat([true_diffs,false_diffs],axis=0)
        diff_cmp = tf.concat([tf.ones_like(true_diffs),tf.zeros_like(false_diffs)],axis=0)

        diff_costs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=all_diffs,labels=diff_cmp))

        minimize_discrim_op = self.discrim_optim.minimize(diff_costs,var_list=self.discrim.vars())

        gen_cost = tf.reduce_mean(-false_diffs)
        minimize_gen_op = self.gen_optim.minimize(gen_cost,var_list=self.gen.vars())

        minimize_op = tf.group([minimize_gen_op,minimize_discrim_op])

        return minimize_op,diff_costs,gen_cost,new_img


def main():
    mc = MainCalc()
    true_img = tf.placeholder(shape=[BATCH_SIZE,96,96,3],dtype=tf.uint8)
    float_img = tf.cast(true_img,tf.float32) / 256.0

    mc_update, diff_l, reconst_l,gen_img = mc.calc_loss(float_img)
    gen_img = tf.cast(gen_img*256.0,tf.uint8)
    # batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(batchnorm_updates)
    # comb_updates = tf.group(batchnorm_updates)
    # tot_update = tf.group([mc_update,comb_updates])

    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    layer_updates = mc.updates()
    print(batchnorm_updates)
    mc_update = tf.group([mc_update]+batchnorm_updates)
    all_l_updates = tf.group(layer_updates)

    orig_datas = []
    full_names =  os.listdir("data/input_data/")[:1000]
    for img_name in full_names:
        with Image.open("data/input_data/"+img_name) as img:
            if img.mode == "RGB":
                arr = np.array(img)
                orig_datas.append((arr))


    out_fold_names = full_names[:50]

    os.makedirs("data/prac_gen_result",exist_ok=True)

    datas = [data for data in orig_datas]
    saver = tf.train.Saver(max_to_keep=50)
    SAVE_DIR = "data/gen_save_model/"
    os.makedirs(SAVE_DIR,exist_ok=True)
    SAVE_NAME = SAVE_DIR+"model.ckpt"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print_num = 0
        lossval_num = 0
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
                random.shuffle(datas)
                tot_diff = 0
                rec_loss = 0
                loss_count = 0
                for data in datas:
                    batch.append(data)
                    if len(batch) == BATCH_SIZE:
                        batch_count += 1
                        img_batch = [img for img in batch]
                        _,dif_l,rec_l = sess.run([mc_update, diff_l, reconst_l],feed_dict={
                            true_img:np.stack(img_batch),
                        })
                        sess.run(all_l_updates)
                        #print(sess.run(float_img))
                        loss_count += 1
                        tot_diff += dif_l
                        rec_loss += rec_l
                        batch = []

                        EPOC_SIZE = 50
                        if batch_count % EPOC_SIZE == 0:
                            print("epoc ended, loss: {}   {}".format(tot_diff/loss_count,rec_loss/loss_count),flush=True)
                            lossval_num += 1

                            tot_diff = 0
                            rec_loss = 0
                            loss_count = 0

                            if batch_count % (EPOC_SIZE*10) == 0:
                                print_num += 1
                                print("save {} started".format(print_num))
                                saver.save(sess,SAVE_NAME,global_step=print_num)
                                batch_outs = sess.run(gen_img)
                                for idx,out in enumerate(batch_outs):
                                    #print(out.shape)
                                    img = Image.fromarray(out)
                                    img_path = "data/prac_gen_result/{}_{}.jpg".format(print_num,idx)
                                    img.save(img_path)
                                print("save {} finished".format(print_num))

if __name__ == "__main__":
    main()
