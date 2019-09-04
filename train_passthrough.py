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
    return (get_out_dim(IMG_SIZE[0],level),get_out_dim(IMG_SIZE[1],level))

def sqr(x):
    return x * x

IMG_LEVEL = 64
SECOND_LEVEL = 96
THIRD_LEVEL = 128
FOURTH_LEVEL = 192
FIFTH_LEVEL = 256
ZIXTH_LEVEL = 256
class MainCalc:
    def __init__(self):
        self.convpool1 = Convpool2(3,IMG_LEVEL,default_activ)
        self.convpool2 = Convpool2(IMG_LEVEL,SECOND_LEVEL,None)
        self.convpool3 = Convpool2(SECOND_LEVEL,THIRD_LEVEL,default_activ)
        self.convpool4 = Convpool2(THIRD_LEVEL,FOURTH_LEVEL,None)
        self.convpool5 = Convpool2(FOURTH_LEVEL,FIFTH_LEVEL,default_activ)
        self.convpool6 = Convpool2(FIFTH_LEVEL,ZIXTH_LEVEL,None)

        self.quanttrans1 = Conv1x1(SECOND_LEVEL,SECOND_LEVEL,None)
        self.quant_block1 = QuantBlockImg(256//2,2,SECOND_LEVEL//2)
        self.quanttrans2 = Conv1x1(FOURTH_LEVEL,FOURTH_LEVEL,None)
        self.quant_block2 = QuantBlockImg(256,4,FOURTH_LEVEL//4)
        self.quant_block3 = QuantBlockImg(256,4,ZIXTH_LEVEL//4)

        self.deconv6 = Deconv2(ZIXTH_LEVEL,FIFTH_LEVEL,default_activ,get_out_shape(6))
        self.deconv5 = Deconv2(FIFTH_LEVEL,FOURTH_LEVEL,default_activ,get_out_shape(5))
        self.deconv4 = Deconv2(FOURTH_LEVEL,THIRD_LEVEL,default_activ,get_out_shape(4))
        self.deconv3 = Deconv2(THIRD_LEVEL,SECOND_LEVEL,default_activ,get_out_shape(3))
        self.deconv2 = Deconv2(SECOND_LEVEL,IMG_LEVEL,default_activ,get_out_shape(2))
        self.deconv1 = Deconv2(IMG_LEVEL,3,tf.sigmoid,get_out_shape(1))

        self.quantupsample32 = Conv1x1Upsample(ZIXTH_LEVEL,FOURTH_LEVEL,None,get_out_shape(5),4)
        self.quantupsample31 = Conv1x1Upsample(ZIXTH_LEVEL,SECOND_LEVEL,None,get_out_shape(3),16)
        self.quantupsample21 = Conv1x1Upsample(FOURTH_LEVEL,SECOND_LEVEL,None,get_out_shape(3),4)

        self.bn1a = tf.layers.BatchNormalization(axis=1)
        self.bn1b = tf.layers.BatchNormalization(axis=1)
        self.bn2a = tf.layers.BatchNormalization(axis=1)
        self.bn2b = tf.layers.BatchNormalization(axis=1)
        self.bn3 = tf.layers.BatchNormalization(axis=1)
        #self.deconvbn2 = tf.layers.BatchNormalization()

    # def all_save_vars(self):
    #     return (
    #         self.conv1.vars("conv1") +
    #         self.conv2.vars("conv2") +
    #         self.conv3.vars("conv3") +
    #         self.conv4.vars("conv4") +
    #         self.conv5.vars("conv5") +
    #         self.conv6.vars("conv6") +
    #         self.quant_block1.vars("qblock1") +
    #         self.quant_block2.vars("qblock2") +
    #         self.quant_block3.vars("qblock3")
    #     )

    def calc(self,input):
        out1 = self.convpool1.calc(input)
        out2 = self.convpool2.calc(out1)
        out3 = self.convpool3.calc(out2)
        out4 = self.convpool4.calc(out3)
        out5 = self.convpool5.calc(out4)
        out6 = self.convpool6.calc(out5)

        quant3,quant_loss3,update3,closest3 = self.quant_block3.calc((self.bn3(out6,training=True)*0.5))
        dec6 = self.deconv6.calc(quant3)
        dec5 = self.deconv5.calc(dec6)
        out4trans = self.quanttrans2.calc(out4)

        quant2,quant_loss2,update2,closest2 = self.quant_block2.calc(((self.bn1a(dec5,training=True)+self.bn1b(out4trans,training=True)))*0.5)
        dec4 = self.deconv4.calc(quant2+self.quantupsample32.calc(quant3))
        dec3 = self.deconv3.calc(dec4)
        out2trans = self.quanttrans1.calc(out2)
        quant1,quant_loss1,update1,closest1 = self.quant_block1.calc(self.bn2a(dec3,training=True)+self.bn2b(out2trans,training=True))
        dec2 = self.deconv2.calc(quant1+self.quantupsample21.calc(quant2)+self.quantupsample31.calc(quant3))
        dec1 = self.deconv1.calc(dec2)
        decoded_final = dec1

        reconstr_loss = tf.reduce_sum(sqr(decoded_final - input))

        quant_loss = quant_loss1 + quant_loss2 + quant_loss3
        tot_loss = reconstr_loss + quant_loss
        tot_update = tf.group([update1,update2,update3])
        closest_list = [closest1,closest2,closest3]
        return tot_update,tot_loss, reconstr_loss,decoded_final,closest_list

    # def __init__(self):
    #     self.convpool1 = Convpool2(3,64,default_activ)
    #     self.convpool2 = Convpool2(64,128,None)
    #     self.quant_block = QuantBlockImg(64,4,32)
    #     self.convunpool1 = Deconv2(128,64,default_activ)
    #     self.convunpool2 = Deconv2(64,3,tf.nn.sigmoid)
    #
    #
    # def calc(self,input):
    #     out1 = self.convpool1.calc(input)
    #     out2 = self.convpool2.calc(out1)
    #     quant,quant_loss,update = self.quant_block.calc(out2)
    #     print(quant.shape)
    #     dec1 = self.convunpool1.calc(quant)
    #     decoded_final = self.convunpool2.calc(dec1)
    #
    #     reconstr_loss = tf.reduce_sum(sqr(decoded_final - input))
    #
    #     tot_loss = reconstr_loss + quant_loss
    #     return update,tot_loss, reconstr_loss,decoded_final

    def periodic_update(self):
        return tf.group([
            self.quant_block1.resample_bad_vecs(),
            self.quant_block2.resample_bad_vecs(),
            self.quant_block3.resample_bad_vecs(),
        ])

def main():
    mc = MainCalc()
    place = tf.placeholder(shape=[BATCH_SIZE,96,96,3],dtype=tf.uint8)
    img_place = tf.cast(place,dtype=tf.float32)/256.0

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    mc_update, loss, reconst_l, final_output,closest_list = mc.calc(img_place)
    resample_update = mc.periodic_update()

    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print(batchnorm_updates)
    comb_updates = tf.group(batchnorm_updates)
    tot_update = tf.group([mc_update,comb_updates])

    opt = optimizer.minimize(loss)
    orig_imgs = []
    orig_filenames = []
    for img_name in os.listdir("data/input_data"):
        with Image.open("data/input_data/"+img_name) as img:
            if img.mode == "RGB":
                arr = np.array(img)
                orig_imgs.append(arr)
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
                random.shuffle(imgs)
                tot_loss = 0
                rec_loss = 0
                loss_count = 0
                for img in imgs:
                    batch.append(img)
                    if len(batch) == BATCH_SIZE:
                        batch_count += 1
                        _,_,cur_loss,cur_rec = sess.run([tot_update,opt,loss,reconst_l],feed_dict={
                            place:np.stack(batch)
                        })
                        loss_count += 1
                        tot_loss += cur_loss
                        rec_loss += cur_rec
                        batch = []

                        EPOC_SIZE = 300
                        if batch_count % EPOC_SIZE == 0:
                            print("epoc ended, loss: {}   {}".format(tot_loss/loss_count,rec_loss/loss_count))
                            lossval_num += 1
                            logfile.write("counts step {} quant 1".format(lossval_num))
                            logfile.write(",".join([str(val.astype(np.int64)) for val in sess.run(mc.quant_block1.vector_counts)])+"\n")
                            logfile.write("counts step {} quant 2".format(lossval_num))
                            logfile.write(",".join([str(val.astype(np.int64)) for val in sess.run(mc.quant_block2.vector_counts)])+"\n")
                            logfile.write("counts step {} quant 3".format(lossval_num))
                            logfile.write(",".join([str(val.astype(np.int64)) for val in sess.run(mc.quant_block3.vector_counts)])+"\n")
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
                                            out = np.transpose(out,(1,2,0))
                                            img = Image.fromarray(out)
                                            img_path = "data/result/{}{}.jpg".format(out_fold,print_num)
                                            #print(img_path)
                                            img.save(img_path)
                                        img_batch = []
                                        fold_batch = []
                                print("save {} finished".format(print_num))

def calc_closest_vals():
    mc = MainCalc()
    place = tf.placeholder(shape=[BATCH_SIZE,96,96,3],dtype=tf.uint8)
    img_place = tf.cast(place,dtype=tf.float32)/256.0

    mc_update, loss, reconst_l, final_output,closest_list = mc.calc(img_place)

    orig_imgs = []
    orig_filenames = []
    for img_name in os.listdir("data/input_data"):
        with Image.open("data/input_data/"+img_name) as img:
            if img.mode == "RGB":
                arr = np.array(img)
                orig_imgs.append(arr)
                orig_filenames.append(img_name)

    full_fold_names = [fname.split('.')[0]+"/" for fname in orig_filenames]

    imgs = orig_imgs
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
        full_saver.restore(sess, checkpoint)
        for idx in range(0,len(imgs)-BATCH_SIZE+1,BATCH_SIZE):
            batch = np.stack(imgs[idx:idx+BATCH_SIZE])

            out_closest_list = sess.run(closest_list,feed_dict={
                place:batch
            })
            closest1,closest2,closest3 = [close.astype(np.uint16) for close in out_closest_list]
            for bidx in range(BATCH_SIZE):
                img_idx = bidx + idx
                new_path = out_path+full_fold_names[img_idx]
                os.mkdir(new_path)
                #image_data = imgs[img_idx]
                #image_data = np.transpose(image_data,(1,2,0))
                #image_data = (image_data*256).astype(np.uint8)
                #np.save(new_path+"image.npy",image_data)
                np.save(new_path+"closest1.npy",closest1[bidx])
                np.save(new_path+"closest2.npy",closest2[bidx])
                np.save(new_path+"closest3.npy",closest3[bidx])

if __name__ == "__main__":
    calc_closest_vals()
