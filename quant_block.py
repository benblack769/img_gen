
import tensorflow as tf
def sqr(x):
    return x * x



def distances(inputs,vecs):
    #return tf.matmul(vecs1,vecs2,transpose_b=True)
    #vecs = tf.transpose(vecs,perm=[1,0,2])
    matmul_val = tf.einsum("ijk,jmk->ijm",inputs,vecs)
    sum_sqr_input = tf.reduce_sum(sqr(inputs), axis=-1, keepdims=True)
    sum_sqr_vecs = tf.reduce_sum(sqr(vecs), axis=-1, keepdims=False)
    dists = (sum_sqr_input
             - 2 * matmul_val
             + sum_sqr_vecs)
    return dists

def gather_multi_idxs(qu_vecs,chosen_idxs):
    idx_shape = chosen_idxs.get_shape().as_list()
    qu_shape = qu_vecs.get_shape().as_list()
    idx_add = tf.range(qu_shape[0],dtype=tf.int64)*qu_shape[1] + chosen_idxs
    idx_transform = tf.reshape(idx_add,[prod(idx_shape)])
    rqu_vecs = tf.reshape(qu_vecs,[qu_shape[0]*qu_shape[1],qu_shape[2]])

    closest_vec_values = tf.gather(rqu_vecs,idx_transform,axis=0)
    #combined_vec_vals = tf.gather(qu_vecs,chosen_idxs,axis=0)

    combined_vec_vals = tf.reshape(closest_vec_values,[idx_shape[0],qu_shape[0]*qu_shape[2]])

    return combined_vec_vals

@tf.custom_gradient
def quant_calc(qu_vecs,chosen_idxs,in_vecs):
    closest_vec_values = gather_multi_idxs(qu_vecs,chosen_idxs)

    def grad(dy):
        return tf.zeros_like(qu_vecs),tf.zeros_like(chosen_idxs),dy
    return closest_vec_values,grad

def assign_moving_average(var,cur_val,decay):
    new_var = var * decay + cur_val * (1-decay)
    print(new_var.shape)
    print(var.shape)
    update = tf.assign(var,new_var)
    return new_var,update


class QuantBlock:
    def __init__(self,QUANT_SIZE,NUM_QUANT,QUANT_DIM):
        init_vals = tf.random_normal([NUM_QUANT,QUANT_SIZE,QUANT_DIM],dtype=tf.float32)
        self.vectors = tf.Variable(init_vals,name="vecs")
        self.vector_counts = tf.Variable(tf.zeros(shape=[NUM_QUANT,QUANT_SIZE],dtype=tf.float32),name="vecs")

        self._ema_cluster_size = tf.Variable(tf.zeros([NUM_QUANT,QUANT_SIZE],dtype=tf.float32))
        #ema_init = tf.reshape(init_vals,[QUANT_SIZE,QUANT_DIM])
        self._ema_w = tf.Variable(init_vals,name='ema_dw')

        self.QUANT_SIZE = QUANT_SIZE
        self.QUANT_DIM = QUANT_DIM
        self.NUM_QUANT = NUM_QUANT
        self._decay = 0.9
        self._epsilon=1e-5

    def calc(self, input):
        orig_size = input.get_shape().as_list()
        div_input = tf.reshape(input,[orig_size[0],self.NUM_QUANT,self.QUANT_DIM])
        #dists = tf.einsum("ijk,jmk->ijm",div_input,self.vectors)
        dists = distances(div_input,self.vectors)
        #cluster_size = tf.reshape(,[self.QUANT_SIZE])
        dists = dists * (1.0+(tf.sqrt(self._ema_cluster_size)))

        #dists = distances(input,self.vectors)
        #soft_vals = tf.softmax(,axis=1)
        #inv_dists = 1.0/(dists+0.000001)
        #closest_vec_idx = tf.multinomial((inv_dists),1)
        #closest_vec_idx = tf.reshape(closest_vec_idx,shape=[closest_vec_idx.get_shape().as_list()[0]])
        #print(closest_vec_idx.shape)
        closest_vec_idx = tf.argmin(dists,axis=-1)

        out_val = quant_calc(self.vectors,closest_vec_idx,input)
        other_losses, update = self.calc_other_vals(input,closest_vec_idx)
        return out_val, other_losses, update,closest_vec_idx

    def codebook_update(self,input,closest_vec_idxs):
        #BATCH_SIZE = input.get_shape().as_list()[0]
        #closest_vec_idxs = tf.reshape(closest_vec_idxs,[BATCH_SIZE,])
        closest_vec_onehots = tf.one_hot(closest_vec_idxs,self.QUANT_SIZE)

        updated_ema_cluster_size,cluster_update = assign_moving_average(
          self._ema_cluster_size, tf.reduce_sum(closest_vec_onehots, axis=0), self._decay)
        dw = tf.einsum("ijk,ijm->jmk",input,closest_vec_onehots)#tf.matmul(input, closest_vec_onehots, transpose_a=True)
        #dw = tf.transpose(dw)
        #print(dw)
        #dw = closest_vec_vals
        #exit(0)
        updated_ema_w,ema_w_update = assign_moving_average(self._ema_w, dw,
                                                            self._decay)
        n = tf.reduce_sum(updated_ema_cluster_size)
        updated_ema_cluster_size = (
          (updated_ema_cluster_size + self._epsilon)
          / (n + self.QUANT_SIZE * self._epsilon) * n)

        normalised_updated_ema_w = (
          updated_ema_w / tf.reshape(updated_ema_cluster_size, [self.NUM_QUANT,self.QUANT_SIZE,1]))
        #update_reshaped = tf.reshape(normalised_updated_ema_w,[self.NUM_QUANT,self.QUANT_SIZE,self.QUANT_DIM])
        update_w = tf.assign(self.vectors, normalised_updated_ema_w)

        all_updates = tf.group([update_w,ema_w_update,cluster_update])
        return all_updates#all_updates

    def calc_other_vals(self,input,closest_vec_idx):
        closest_vec_values = gather_multi_idxs(self.vectors,closest_vec_idx)

        #codebook_loss = tf.reduce_sum(sqr(closest_vec_values - tf.stop_gradient(input)))
        orig_size = input.get_shape().as_list()
        div_input = tf.reshape(input,[orig_size[0],self.NUM_QUANT,self.QUANT_DIM])
        codebook_update = self.codebook_update(div_input,closest_vec_idx)

        beta_val = 0.25 #from https://arxiv.org/pdf/1906.00446.pdf
        commitment_loss = tf.reduce_sum(beta_val * sqr(tf.stop_gradient(closest_vec_values) - input))

        idx_one_hot = tf.one_hot(closest_vec_idx,self.QUANT_SIZE)
        total = tf.reduce_sum(idx_one_hot,axis=0)
        update_counts = tf.assign(self.vector_counts,self.vector_counts+total)
        combined_update = tf.group([codebook_update,update_counts])

        return commitment_loss ,combined_update

    def resample_bad_vecs(self):
        sample_vals = tf.random_normal([self.NUM_QUANT,self.QUANT_SIZE,self.QUANT_DIM],dtype=tf.float32)
        equal_vals = tf.cast(tf.equal(self.vector_counts,0),dtype=tf.float32)
        equal_vals= tf.reshape(equal_vals,shape=[self.NUM_QUANT,self.QUANT_SIZE,1])
        new_vecs = self.vectors - self.vectors * equal_vals + sample_vals * equal_vals
        vec_assign = tf.assign(self.vectors,new_vecs)
        ema_assign = tf.assign(self._ema_w,new_vecs)
        zero_assign = tf.assign(self.vector_counts,tf.zeros_like(self.vector_counts))
        tot_assign = tf.group([vec_assign,zero_assign,ema_assign])
        return zero_assign#tot_assign

    def vars(self,name):
        return [
            (name+"vecs",self.vectors),
            (name+"emq_w",self._ema_cluster_size),
            (name+"vecs",self._ema_w),
        ]

def prod(l):
    p = 1
    for x in l:
        p *= x
    return p

class QuantBlockImg(QuantBlock):
    def calc(self,input):
        input = tf.transpose(input,perm=(0,2,3,1))
        in_shape = input.get_shape().as_list()
        flat_val = tf.reshape(input,[prod(in_shape[:3]),in_shape[3]])
        out,o1,o2,closest = QuantBlock.calc(self,flat_val)
        restored = tf.reshape(out,in_shape)
        restored = tf.transpose(restored,perm=(0,3,1,2))
        resh_closest = tf.reshape(closest,[in_shape[0],in_shape[1],in_shape[2],closest.shape[1]])
        return restored,o1,o2,resh_closest
