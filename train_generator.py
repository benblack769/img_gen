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

def prod(l):
    p = 1
    for x in l:
        p *= x
    return p

def full_flatten(tens4d):
    size = prod(tens4d.get_shape().as_list())
    return tf.reshape(tens4d,[size])

def calc_diff(tens4d):
    shape = tens4d.get_shape().as_list()
    dived = tf.reshape(tens4d,shape[:3]+[2,shape[3]//2])
    mapped = tf.math.reduce_prod(dived,axis=3)
    print(mapped.shape)
    print(mapped.shape)
    print(mapped.shape)
    print(mapped.shape)
    print(mapped.shape)
    summed = tf.reduce_mean(mapped)*(shape[3]//2)
    return full_flatten(summed)

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

        self.convpool3repr = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ)
        self.convpool4repr = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,default_activ)

        self.diff1img = Conv1x1(IMG_LEVEL,IMG_LEVEL*2,None)
        self.diff2img = Conv1x1(SECOND_LEVEL,SECOND_LEVEL*2,None)
        #self.diff3img = Conv1x1(THIRD_LEVEL,THIRD_LEVEL*2,None)
        #self.diff4img = Conv1x1(FOURTH_LEVEL,FOURTH_LEVEL*2,None)
        self.gather3img = Conv1x1(THIRD_LEVEL,THIRD_LEVEL,None)
        self.gather4img = Conv1x1(FOURTH_LEVEL,FOURTH_LEVEL,None)

        self.gather3repr = Conv1x1(THIRD_LEVEL,THIRD_LEVEL,None)
        self.gather4repr = Conv1x1(FOURTH_LEVEL,FOURTH_LEVEL,None)


    def updates(self):
        return (
            self.convpool1img.updates() +
            self.convpool2img.updates() +
            self.convpool3img.updates() +
            self.convpool4img.updates() +
            self.convpool3repr.updates() +
            self.convpool4repr.updates()
        )

    def calc(self,img,repr):
        cur_img_out = img
        cur_img_out = self.convpool1img.calc(cur_img_out)
        diff1 = self.diff1img.calc(cur_img_out)
        cur_img_out = self.convpool2img.calc(cur_img_out)
        diff2 = self.diff2img.calc(cur_img_out)
        cur_img_out = self.convpool3img.calc(cur_img_out)
        #diff3 = self.diff3img.calc(cur_img_out)
        l3img = self.gather3img.calc(cur_img_out)
        cur_img_out = self.convpool4img.calc(cur_img_out)
        #diff4 = self.diff4img.calc(cur_img_out)
        l4img = self.gather4img.calc(cur_img_out)

        cur_repr_out = repr
        cur_repr_out = self.convpool3repr.calc(cur_repr_out)
        repr3out = self.gather3repr.calc(cur_repr_out)
        cur_repr_out = self.convpool4repr.calc(cur_repr_out)
        repr4out = self.gather4repr.calc(cur_repr_out)

        diff3 = (tf.reduce_mean(l3img * repr3out))*THIRD_LEVEL
        diff4 = (tf.reduce_mean(l4img * repr4out))*FOURTH_LEVEL
        #d1 = (calc_diff(diff1))
        #d2 = (calc_diff(diff2))
        fin_loss =  diff3+diff4#+d1+d2
        return tf.reshape(fin_loss,[1])

    def vars(self):
        var_names = (
            self.convpool1img.vars("") +
            self.convpool2img.vars("") +
            self.convpool3img.vars("") +
            self.convpool4img.vars("") +
            self.convpool3repr.vars("") +
            self.convpool4repr.vars("") +
            self.gather3img.vars("") +
            self.gather4img.vars("") +
            self.gather3repr.vars("") +
            self.gather4repr.vars("") +
            self.diff1img.vars("") +
            self.diff2img.vars("")
        )
        vars = [var for name, var in var_names]
        return vars

class Gen:
    def __init__(self):
        self.RAND_SIZE = RAND_SIZE = 4
        self.convpool1 = Convpool2(6,IMG_LEVEL,default_activ)
        self.convpool2 = Convpool2(IMG_LEVEL+RAND_SIZE,SECOND_LEVEL,default_activ)
        self.convpool3 = Convpool2(SECOND_LEVEL+RAND_SIZE,THIRD_LEVEL,default_activ)
        self.convpool4 = Convpool2(THIRD_LEVEL+RAND_SIZE,FOURTH_LEVEL,default_activ)

        self.deconv4 = Deconv2(FOURTH_LEVEL,THIRD_LEVEL,default_activ,get_out_shape(4))
        self.deconv3 = Deconv2(THIRD_LEVEL,SECOND_LEVEL,default_activ,get_out_shape(3))
        self.deconv2 = Deconv2(SECOND_LEVEL,IMG_LEVEL,default_activ,get_out_shape(2))
        self.deconv1 = Deconv2(IMG_LEVEL,3,tf.sigmoid,get_out_shape(1))

        self.out_trans3 = Conv1x1(THIRD_LEVEL,THIRD_LEVEL,default_activ)
        self.out_trans2 = Conv1x1(SECOND_LEVEL,SECOND_LEVEL,default_activ)
        self.out_trans1 = Conv1x1(IMG_LEVEL,IMG_LEVEL,default_activ)

        self.repr_vecs = Conv1x1(SECOND_LEVEL,SECOND_LEVEL,None)

    def updates(self):
        return (
            self.convpool1.updates() +
            self.convpool2.updates() +
            self.convpool3.updates() +
            self.convpool4.updates() +
            self.deconv1.updates() +
            self.deconv2.updates() +
            self.deconv3.updates() +
            self.deconv4.updates() +
            self.out_trans1.updates() +
            self.out_trans2.updates() +
            self.out_trans3.updates()
        )

    def calc(self, old_img, repr):
        repr_v = self.repr_vecs.calc(repr)
        rand_l2 = tf.random.normal([BATCH_SIZE]+get_out_shape(2)+[self.RAND_SIZE])
        rand_l3 = tf.random.normal([BATCH_SIZE]+get_out_shape(3)+[self.RAND_SIZE])
        rand_l4 = tf.random.normal([BATCH_SIZE]+get_out_shape(4)+[self.RAND_SIZE])

        comb1 = self.convpool1.calc(old_img)
        comb2 = repr_v + self.convpool2.calc(tf.concat([comb1,rand_l2],axis=3))
        comb3 = self.convpool3.calc(tf.concat([comb2,rand_l3],axis=3))
        comb4 = self.convpool4.calc(tf.concat([comb3,rand_l4],axis=3))

        deconv4 = self.deconv4.calc(comb4)
        deconv3 = self.deconv3.calc(self.out_trans3.calc(comb3) + deconv4)
        deconv2 = self.deconv2.calc(self.out_trans2.calc(comb2) + deconv3)
        deconv1 = self.deconv1.calc(self.out_trans1.calc(comb1) + deconv2)

        fin_out = deconv1

        return fin_out

    def vars(self):
        var_names = (
            self.convpool1.vars("") +
            self.convpool2.vars("") +
            self.convpool3.vars("") +
            self.convpool4.vars("") +
            self.deconv4.vars("") +
            self.deconv3.vars("") +
            self.deconv2.vars("") +
            self.deconv1.vars("") +
            self.out_trans3.vars("") +
            self.out_trans2.vars("") +
            self.out_trans1.vars("") +
            self.repr_vecs.vars("")
        )
        vars = [var for name, var in var_names]
        return vars

class MainCalc:
    def __init__(self):
        self.gen = Gen()
        self.discrim = Discrim()
        self.discrim_optim = tf.train.RMSPropOptimizer(learning_rate=0.00001,decay=0.9)
        self.gen_optim = tf.train.RMSPropOptimizer(learning_rate=0.00001,decay=0.9)
        self.bn_grads = tf.layers.BatchNormalization(axis=3)

    def updates(self):
        return self.discrim.updates() #+ self.gen.updates()

    def calc_loss(self,true_imgs,old_img,repr_idxs):
        REPR_SIZE = 2
        REPR_DEPTH = 128
        repr = tf.one_hot(repr_idxs,depth=REPR_DEPTH)
        repr = tf.reshape(repr,[BATCH_SIZE]+get_out_shape(3)+[REPR_DEPTH*REPR_SIZE])

        new_img = self.gen.calc(old_img,repr)

        true_diffs = self.discrim.calc(true_imgs,repr)
        false_diffs = self.discrim.calc(new_img,repr)

        all_diffs = tf.concat([true_diffs,false_diffs],axis=0)
        diff_cmp = tf.concat([tf.ones_like(true_diffs),tf.zeros_like(false_diffs)],axis=0)

        diff_costs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=all_diffs,labels=diff_cmp))

        reconstr_l = 0.1*tf.reduce_mean(sqr(new_img - true_imgs))
        disting_costs = -tf.reduce_mean(false_diffs)

        minimize_discrim_op = self.discrim_optim.minimize(diff_costs,var_list=self.discrim.vars())
        minimize_gen_op = self.gen_optim.minimize(disting_costs,var_list=self.gen.vars())

        minimize_op = tf.group([minimize_gen_op,minimize_discrim_op])

        new_img_grad = tf.ones_like(new_img)#tf.gradients(ys=false_diffs,xs=new_img,stop_gradients=[old_img,repr_idxs])[0]
        #new_img_grad = tf.stop_gradient(self.bn_grads(new_img_grad))

        return minimize_op,tf.stop_gradient(new_img),new_img_grad,reconstr_l,disting_costs,diff_costs

    def recursive_calc(self,true_imgs,repr_idxs):
        cur_old_img = tf.concat([tf.zeros_like(true_imgs),tf.ones_like(true_imgs)],axis=3)
        all_reconstr_l = tf.zeros(1)
        all_diff_l = tf.zeros(1)
        all_disting_l = tf.zeros(1)

        all_updates = []
        for x in range(1):
            minimize_op,new_img,new_img_grad,reconstr_l,disting_l,diff_costs = self.calc_loss(true_imgs,cur_old_img,repr_idxs)
            cur_old_img = tf.concat([new_img,new_img_grad],axis=3)
            all_reconstr_l += reconstr_l
            all_diff_l += diff_costs
            all_disting_l += disting_l
            all_updates.append(minimize_op)
            if x == 0:
                first_new_img = new_img

        return tf.group(all_updates),all_diff_l[0],all_disting_l[0],all_reconstr_l[0],first_new_img


def main():
    mc = MainCalc()
    true_img = tf.placeholder(shape=[BATCH_SIZE,96,96,3],dtype=tf.uint8)
    float_img = tf.cast(true_img,tf.float32) / 256.0
    cmp_idxs = tf.placeholder(shape=[BATCH_SIZE]+get_out_shape(3)+[2],dtype=tf.uint16)
    cmp_idx32 = tf.cast(cmp_idxs,tf.int32)

    mc_update, diff_l, disting_l, reconst_l, final_img = mc.recursive_calc(float_img,cmp_idx32)
    # batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(batchnorm_updates)
    # comb_updates = tf.group(batchnorm_updates)
    # tot_update = tf.group([mc_update,comb_updates])

    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print(batchnorm_updates)
    layer_updates = mc.updates()
    mc_update = tf.group([mc_update]+batchnorm_updates+layer_updates)

    orig_datas = []
    full_names =  os.listdir("data/pretrained_result/")
    for img_name in full_names:
        with Image.open("data/input_data/"+img_name+".jpg") as img:
            #if img.mode == "RGB":
            arr = np.array(img)
            repr = np.load("data/pretrained_result/"+img_name+"/closest1.npy")
            orig_datas.append((arr,repr))


    out_fold_names = full_names[:max(BATCH_SIZE,50)]

    for fold,fname in zip(out_fold_names,full_names):
        fold_path = "data/gen_result/"+fold + "/"
        os.makedirs(fold_path,exist_ok=True)
        shutil.copy("data/input_data/"+fname+".jpg",fold_path+"orig.jpg")

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
                tot_dis = 0
                rec_loss = 0
                loss_count = 0
                for data in datas:
                    batch.append(data)
                    if len(batch) == BATCH_SIZE:
                        batch_count += 1
                        img_batch = [img for img,repr in batch]
                        repr_batch = [repr for img,repr in batch]
                        _,dif_l,dis_l,rec_l = sess.run([mc_update, diff_l, disting_l, reconst_l],feed_dict={
                            true_img:np.stack(img_batch),
                            cmp_idxs:np.stack(repr_batch)
                        })
                        #print(sess.run(float_img))
                        loss_count += 1
                        tot_diff += dif_l
                        tot_dis += dis_l
                        rec_loss += rec_l
                        batch = []

                        EPOC_SIZE = 100
                        if batch_count % EPOC_SIZE == 0:
                            print("epoc ended, loss: {}   {}    {}".format(tot_diff/loss_count,rec_loss/loss_count,tot_dis/loss_count),flush=True)
                            lossval_num += 1

                            tot_diff = 0
                            tot_dis = 0
                            rec_loss = 0
                            loss_count = 0

                            if batch_count % (EPOC_SIZE*10) == 0:
                                print_num += 1
                                print("save {} started".format(print_num))
                                saver.save(sess,SAVE_NAME,global_step=print_num)
                                data_batch = []
                                fold_batch = []
                                for count,(data,fold) in enumerate(zip(orig_datas,out_fold_names)):
                                    data_batch.append((data))
                                    fold_batch.append((fold))
                                    if len(data_batch) == BATCH_SIZE:

                                        img_batch = [img for img,repr in data_batch]
                                        repr_batch = [repr for img,repr in data_batch]
                                        batch_outs = sess.run(final_img,feed_dict={
                                            true_img:np.stack(img_batch),
                                            cmp_idxs:np.stack(repr_batch)
                                        })
                                        pixel_vals = (batch_outs * 256).astype(np.uint8)
                                        for out,out_fold in zip(pixel_vals,fold_batch):
                                            #print(out.shape)
                                            img = Image.fromarray(out)
                                            img_path = "data/gen_result/{}/{}.jpg".format(out_fold,print_num)
                                            #print(img_path)
                                            img.save(img_path)
                                        data_batch = []
                                        fold_batch = []
                                print("save {} finished".format(print_num))

if __name__ == "__main__":
    main()
