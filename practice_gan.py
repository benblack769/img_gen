import os
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import shutil
from base_ops import default_activ,Convpool2,Conv2d,Conv1x1,Conv1x1Upsample,ConvTrans2d,Convpool2,Deconv2,Dense
from quant_block import QuantBlockImg
from npy_saver import NpySaver

BATCH_SIZE = 16
BATCHS_PER_UPDATE = 8
UPDATE_COUNT = 4

IMG_SIZE = (96,96)

def round_up_div(num,denom):
    return (num+denom-1) // denom

def get_out_dim(dim,level):
    return round_up_div(dim, 2**(level-1))

def get_out_shape(level):
    return [get_out_dim(IMG_SIZE[0],level),get_out_dim(IMG_SIZE[1],level)]

def broadcast_shape(tens,level):
    level_shape = get_out_shape(level)
    tiled = tf.tile(tens,[1]+level_shape+[1])
    return tiled

def prod(l):
    p = 1
    for v in l:
        p *= v
    return p

def sqr(x):
    return x * x

IMG_LEVEL = 32
SECOND_LEVEL = 64
THIRD_LEVEL = 128
FOURTH_LEVEL = 192
FIFTH_LEVEL = 256
SIXTH_LEVEL = 256

FLAT_LEVEL = 512

class Discrim:
    def __init__(self):
        self.convpool1img = Convpool2(3,IMG_LEVEL,default_activ)
        self.convpool2img = Convpool2(IMG_LEVEL,SECOND_LEVEL,default_activ)
        self.convpool3img = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ)
        self.convpool4img = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,default_activ)
        self.convpool5img = Convpool2(FOURTH_LEVEL,FIFTH_LEVEL,default_activ)
        self.convpool6img = Convpool2(FIFTH_LEVEL,SIXTH_LEVEL,default_activ)
        #self.denseout1 = Dense(SIXTH_LEVEL*prod(get_out_shape(6)),FLAT_LEVEL,default_activ)
        #self.denseout2 = Dense(FLAT_LEVEL,1,None)

        self.comb_assess = Conv1x1(SIXTH_LEVEL,1,None)

    def calc(self,img):
        cur_img_out = img
        cur_img_out = self.convpool1img.calc(cur_img_out)
        cur_img_out = self.convpool2img.calc(cur_img_out)
        cur_img_out = self.convpool3img.calc(cur_img_out)
        cur_img_out = self.convpool4img.calc(cur_img_out)
        cur_img_out = self.convpool5img.calc(cur_img_out)
        cur_img_out = self.convpool6img.calc(cur_img_out)
        cur_img_out = self.comb_assess.calc(cur_img_out)
        #cur_img_out = tf.reshape(cur_img_out,[BATCH_SIZE,SIXTH_LEVEL*prod(get_out_shape(6))])
        #cur_img_out = self.denseout1.calc(cur_img_out)
        #cur_img_out = self.denseout2.calc(cur_img_out)

        return cur_img_out

    def updates(self):
        return (
            self.convpool1img.updates() +
            self.convpool2img.updates() +
            self.convpool3img.updates() +
            self.convpool4img.updates() +
            self.convpool5img.updates() +
            self.convpool6img.updates() +
            self.comb_assess.updates() #+
            #self.denseout1.updates() +
            #self.denseout2.updates() +
            #self.comb_assess.updates()
        )

    def vars(self):
        var_names = (
            self.convpool1img.vars("") +
            self.convpool2img.vars("") +
            self.convpool3img.vars("") +
            self.convpool4img.vars("") +
            self.convpool5img.vars("") +
            self.convpool6img.vars("") +
            self.comb_assess.vars("")
        )
        vars = [var for name, var in var_names]
        return vars

RAND_SIZE = 32
HINT_SIZE = 16
class Gen:
    def __init__(self):
        self.convpool1hint = Convpool2(6,IMG_LEVEL,default_activ)
        self.convpool2hint = Convpool2(IMG_LEVEL,SECOND_LEVEL,default_activ)
        self.convpool3hint = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ)
        self.convpool4hint = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,default_activ)
        self.convpool5hint = Convpool2(FOURTH_LEVEL,FIFTH_LEVEL,default_activ)
        self.convpool6hint = Convpool2(FIFTH_LEVEL,SIXTH_LEVEL,default_activ)

        self.hint_condenser1 = Conv1x1(IMG_LEVEL,HINT_SIZE,None)
        self.hint_condenser2 = Conv1x1(SECOND_LEVEL,HINT_SIZE,None)
        self.hint_condenser3 = Conv1x1(THIRD_LEVEL,HINT_SIZE,None)
        self.hint_condenser4 = Conv1x1(FOURTH_LEVEL,HINT_SIZE,None)
        self.hint_condenser5 = Conv1x1(FIFTH_LEVEL,HINT_SIZE,None)
        self.hint_condenser6 = Conv1x1(SIXTH_LEVEL,HINT_SIZE,None)


        self.deconv6 = Deconv2(RAND_SIZE+HINT_SIZE,FIFTH_LEVEL,default_activ,get_out_shape(6))
        self.deconv5 = Deconv2(RAND_SIZE+HINT_SIZE+FIFTH_LEVEL,FOURTH_LEVEL,default_activ,get_out_shape(5))
        self.deconv4 = Deconv2(RAND_SIZE+HINT_SIZE+FOURTH_LEVEL,THIRD_LEVEL,default_activ,get_out_shape(4))
        self.deconv3 = Deconv2(RAND_SIZE+HINT_SIZE+THIRD_LEVEL,SECOND_LEVEL,default_activ,get_out_shape(3))
        self.deconv2 = Deconv2(RAND_SIZE+HINT_SIZE+SECOND_LEVEL,IMG_LEVEL,default_activ,get_out_shape(2))
        self.deconv1 = Deconv2(IMG_LEVEL+HINT_SIZE,3,tf.sigmoid,get_out_shape(1))

        self.hint_batchnorm = tf.layers.BatchNormalization()

    def calc(self,hint_img):
        hint_img = self.hint_batchnorm(hint_img)
        hint_data1 = self.convpool1hint.calc(hint_img)
        hint_data2 = self.convpool2hint.calc(hint_data1)
        hint_data3 = self.convpool3hint.calc(hint_data2)
        hint_data4 = self.convpool4hint.calc(hint_data3)
        hint_data5 = self.convpool5hint.calc(hint_data4)
        hint_data6 = self.convpool6hint.calc(hint_data5)

        cond_hint1 = self.hint_condenser1.calc(hint_data1)
        cond_hint2 = self.hint_condenser2.calc(hint_data2)
        cond_hint3 = self.hint_condenser3.calc(hint_data3)
        cond_hint4 = self.hint_condenser4.calc(hint_data4)
        cond_hint5 = self.hint_condenser5.calc(hint_data5)
        cond_hint6 = self.hint_condenser6.calc(hint_data6)

        def attach_rand(tens,hint,level):
            return tf.concat([tens,hint,broadcast_shape(rand_inp,level)],axis=-1)

        rand_inp = tf.random.normal(shape=[BATCH_SIZE,1,1,RAND_SIZE])
        tiled_rand_inp = broadcast_shape(rand_inp,7)

        deconv6 = self.deconv6.calc(tf.concat([tiled_rand_inp,cond_hint6],axis=-1))
        deconv5 = self.deconv5.calc(attach_rand(deconv6,cond_hint5,6))
        deconv4 = self.deconv4.calc(attach_rand(deconv5,cond_hint4,5))
        deconv3 = self.deconv3.calc(attach_rand(deconv4,cond_hint3,4))
        deconv2 = self.deconv2.calc(attach_rand(deconv3,cond_hint2,3))
        deconv1 = self.deconv1.calc(tf.concat([deconv2,cond_hint1],axis=-1))

        fin_out = deconv1

        return fin_out

    def updates(self):
        return (
            self.convpool1hint.updates() +
            self.convpool2hint.updates() +
            self.convpool3hint.updates() +
            self.convpool4hint.updates() +
            self.convpool5hint.updates() +
            self.convpool6hint.updates() +

            self.hint_condenser1.updates() +
            self.hint_condenser2.updates() +
            self.hint_condenser3.updates() +
            self.hint_condenser4.updates() +
            self.hint_condenser5.updates() +
            self.hint_condenser6.updates() +

            self.deconv6.updates() +
            self.deconv5.updates() +
            self.deconv4.updates() +
            self.deconv3.updates() +
            self.deconv2.updates() +
            self.deconv1.updates()
        )

    def vars(self):
        var_names = (
            self.convpool1hint.vars("") +
            self.convpool2hint.vars("") +
            self.convpool3hint.vars("") +
            self.convpool4hint.vars("") +
            self.convpool5hint.vars("") +
            self.convpool6hint.vars("") +

            self.hint_condenser1.vars("") +
            self.hint_condenser2.vars("") +
            self.hint_condenser3.vars("") +
            self.hint_condenser4.vars("") +
            self.hint_condenser5.vars("") +
            self.hint_condenser6.vars("") +

            self.deconv6.vars("") +
            self.deconv5.vars("") +
            self.deconv4.vars("") +
            self.deconv3.vars("") +
            self.deconv2.vars("") +
            self.deconv1.vars("")
        )
        vars = [var for name, var in var_names]
        return vars


def add_gradients(gradvar_list):
    add_updates = []
    var_gradvar_list = []
    initialize_updates = []
    for grad,var in gradvar_list:
        gradvar = tf.Variable(tf.zeros_like(grad),dtype=tf.float32)
        var_gradvar_list.append((gradvar,var))
        add_updates.append(tf.assign(gradvar,gradvar+grad))
        initialize_updates.append(tf.assign(gradvar,tf.zeros_like(grad)))

    return var_gradvar_list,tf.group(add_updates),tf.group(initialize_updates)

def minimize_over_updates(optimizer,gradvar_list):
    added_gradvars,add_op,init_op = add_gradients(gradvar_list)
    return optimizer.apply_gradients(added_gradvars),add_op,init_op

class MainCalc:
    def __init__(self):
        self.gen = Gen()
        self.discrim = Discrim()
        self.discrim_optim = tf.train.RMSPropOptimizer(learning_rate=0.00001,decay=0.9)
        self.gen_optim = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9)
        self.bn_grads = tf.layers.BatchNormalization(axis=1)

    def updates(self):
        return  self.discrim.updates()+self.gen.updates()

    def calc_loss_single(self,true_imgs,hint_img):
        new_img = self.gen.calc(hint_img)

        true_diffs = self.discrim.calc(true_imgs)
        false_diffs = self.discrim.calc(new_img)

        all_diffs = tf.concat([true_diffs,false_diffs],axis=0)
        diff_cmp = tf.concat([tf.ones_like(true_diffs),tf.zeros_like(false_diffs)],axis=0)

        diff_costs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=all_diffs,labels=diff_cmp))

        gen_cost = tf.reduce_mean(-false_diffs)

        new_img_grad = tf.gradients(ys=gen_cost,xs=new_img,stop_gradients=[hint_img])[0]

        return diff_costs,gen_cost,tf.stop_gradient(new_img),tf.stop_gradient(new_img_grad)

    def calc_updates(self,true_imgs):
        cur_hint = tf.zeros([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],6])

        all_diff_costs = []
        all_gen_costs = []
        for i in range(UPDATE_COUNT):
            diff_costs,gen_cost,new_img,new_img_grad = self.calc_loss_single(true_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE],cur_hint)
            all_diff_costs.append(diff_costs)
            all_gen_costs.append(gen_cost)
            cur_hint = tf.concat([new_img,new_img_grad],axis=-1)

        diff_costs = tf.math.accumulate_n(all_diff_costs)/UPDATE_COUNT
        gen_cost = tf.math.accumulate_n(all_gen_costs)/UPDATE_COUNT

        discrim_grads = self.discrim_optim.compute_gradients(diff_costs,var_list=self.discrim.vars())
        discrim_update_apply,discrim_update_add,discrim_update_init = minimize_over_updates(self.discrim_optim,discrim_grads)
        #minimize_gen_op = self.gen_optim.minimize(gen_cost,var_list=self.gen.vars())
        gen_grads = self.gen_optim.compute_gradients(diff_costs,var_list=self.gen.vars())
        gen_update_apply,gen_update_add,gen_update_init = minimize_over_updates(self.gen_optim,gen_grads)


        apply_op = tf.group([discrim_update_apply,gen_update_apply])
        add_op = tf.group([discrim_update_add,gen_update_add])
        init_op = tf.group([discrim_update_init,gen_update_init])

        return apply_op,add_op,init_op,diff_costs,gen_cost,new_img

        #n_apply_op,n_add_op,n_init_op,n_diff_costs,n_gen_cost,n_new_img = self.calc_loss_single(true_imgs,orig_hint)


def main():
    mc = MainCalc()
    true_img = tf.placeholder(shape=[BATCH_SIZE*UPDATE_COUNT,96,96,3],dtype=tf.uint8)
    float_img = tf.cast(true_img,tf.float32) / 256.0

    apply_op,add_op,init_op, diff_l, reconst_l,gen_img = mc.calc_updates(float_img)
    gen_img = tf.cast(gen_img*256.0,tf.uint8)
    # batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(batchnorm_updates)
    # comb_updates = tf.group(batchnorm_updates)
    # tot_update = tf.group([mc_update,comb_updates])

    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    layer_updates = mc.updates()
    print(batchnorm_updates)
    apply_op = tf.group([apply_op]+batchnorm_updates)
    all_l_updates = tf.group(layer_updates)

    orig_datas = []
    full_names =  os.listdir("data/input_data/")
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
        update_count = 0
        while True:
            for x in range(20):
                random.shuffle(datas)
                tot_diff = 0
                rec_loss = 0
                loss_count = 0
                for data in datas:
                    batch.append(data)
                    if len(batch) == BATCH_SIZE*UPDATE_COUNT:
                        batch_count += 1
                        img_batch = batch
                        if batch_count % BATCHS_PER_UPDATE != 0:

                            _ = sess.run(add_op,feed_dict={
                                true_img:np.stack(img_batch),
                            })
                            batch = []
                        else:
                            update_count += 1
                            _,dif_l,rec_l = sess.run([apply_op, diff_l, reconst_l],feed_dict={
                                true_img:np.stack(img_batch),
                            })
                            sess.run(all_l_updates)
                            sess.run(init_op)
                            #print(sess.run(float_img))
                            loss_count += 1
                            tot_diff += dif_l
                            rec_loss += rec_l
                            batch = []

                            EPOC_SIZE = 50
                            if update_count % EPOC_SIZE == 0:
                                print("epoc ended, loss: {}   {}".format(tot_diff/loss_count,rec_loss/loss_count),flush=True)
                                lossval_num += 1

                                tot_diff = 0
                                rec_loss = 0
                                loss_count = 0

                                if update_count % (EPOC_SIZE*10) == 0:
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
