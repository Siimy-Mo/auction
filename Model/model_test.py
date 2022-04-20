import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Concatenate,Dense
import numpy as np
import math
import six
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

UW=0.08
IW=0.06
MAX_LEN = 10
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1

class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head # 128/4 = 32
        self.n_head = n_head    # 4
        self.model_dim = model_dim  # 128
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(n_head * self.head_dim)      # [n, step, h*h_dim]

        self.o_dense = keras.layers.Dense(model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def call(self, q, k, v, mask, training):
        q = tf.reshape(q, (tf.shape(q)[0], tf.shape(q)[1], 120))
        _q = self.wq(q)      # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)     # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)     # [n, q_step, h*dv]
        o = self.o_dense(context)       # [n, step, dim]
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])       # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)                               # [n, h, q_step, step]
        context = tf.matmul(self.attention, v)         # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])   # [n, q_step, h, dv]
        context = tf.reshape(context, (tf.shape(context)[0], tf.shape(context)[1], self.n_head* self.head_dim))     # [n, q_step, h*dv] -1初始化的时候dim= None会报错
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o         # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)]  # only norm z-dim
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, td, training, mask):
        attn = self.mh.call(xz, xz, xz, mask, training)       # [n, step, dim]
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.ffn.call(o1), training)
        o = self.ln[1](ffn + o1)         # [n, step, dim]
        return o

class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, xz, td,training, mask):
        for l in self.ls:
            xz = l.call(xz,td, training, mask)
        return xz       # [n, step, dim]

class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(3)] # only norm z-dim
        self.drop = keras.layers.Dropout(drop_rate)
        self.mh = [MultiHead(n_head, model_dim, drop_rate) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        attn = self.mh[0].call(yz, yz, yz, yz_look_ahead_mask, training)       # decoder self attention
        o1 = self.ln[0](attn + yz)
        attn = self.mh[1].call(o1, xz, xz, xz_pad_mask, training)       # decoder + encoder attention
        o2 = self.ln[1](attn + o1)
        ffn = self.drop(self.ffn.call(o2), training)
        o = self.ln[2](ffn + o2)
        return o

class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for l in self.ls:
            yz = l.call(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim):    # n_vocab的用处。
        super().__init__()
        pos = np.arange(max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim)  # [max_len, dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]  # [1, max_len, model_dim]    for batch adding
        self.pe = tf.constant(pe, dtype=tf.float32)


    def call(self, Embed):
        # x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        x_embed = Embed + self.pe  # [n, step, dim]
        return x_embed


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.embed = PositionEmbedding(max_len, model_dim, n_vocab)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(n_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(0.002)

    def call(self, x, y, training=None):
        x_embed, y_embed = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder.call(x_embed, training, mask=pad_mask)
        decoded_z = self.decoder.call(
            y_embed, encoded_z, training, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=pad_mask)
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.call(x, y[:, :-1], training=True)
            pad_mask = tf.math.not_equal(y[:, 1:], self.padding_idx)
            loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(y[:, 1:], logits), pad_mask))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, logits

    def _pad_bool(self, seqs):
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self, seqs):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        mask = tf.where(self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.mh[0].attention.numpy() for l in self.decoder.ls],
                "mh2": [l.mh[1].attention.numpy() for l in self.decoder.ls],
        }}
        return attentions




def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    # print(loss, loss.shape)
    # print(vars)
    # for i in vars:
    #     print(i.name,'with shape:',i.shape)
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.compat.v1.summary.histogram(variable.name, variable)
            tf.compat.v1.summary.histogram(variable.name + '/gradients', grad_values)
            tf.compat.v1.summary.histogram(
                variable.name + '/gradient_norm',
                tf.linalg.global_norm([grad_values])
                # clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step) #更新variable

def Loss_F(useremb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb    # 2d
    dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis = 1), -1.0)
    log_u_plus_q_minus_i_pos = tf.math.log(tf.sigmoid(dis_pos))
    
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    log_u_plus_q_minus_i_neg = tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg)),axis=1)
    batch_loss = -1 * (log_u_plus_q_minus_i_neg + log_u_plus_q_minus_i_pos)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(-dis_pos), tf.reduce_mean(dis_neg)

def Inner_product(useremb, product_pos, product_neg, k):
    # u_plus_q = user+query
    u_plus_q = useremb # 64 
    dis_pos = tf.reduce_sum(tf.multiply(u_plus_q, product_pos), axis=1) 
    loss_pos = tf.reduce_mean(tf.math.log(tf.sigmoid(dis_pos)))

    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1]) # 64 10
    dis_neg = tf.reduce_sum(tf.multiply(expand_u_plus_q, product_neg), axis=2) 
    loss_neg = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.sigmoid(tf.multiply(dis_neg, -1.0))),axis=1))
    
    batch_loss = -1.0*(loss_pos + loss_neg)
    #return tf.reduce_mean(batch_loss), tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)

def Pairwise_Loss_F(useremb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb 
    dis_pos = tf.norm(u_plus_q - product_pos, ord=2, axis=1)
    dis_pos = tf.reshape(dis_pos, [-1, 1])


    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1, k, 1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    
    batch_loss = tf.multiply(tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg - k * dis_pos) + 1e-6), axis=1), -1.0)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)

tf.compat.v1.disable_eager_execution()

class Seq(object):
    def __init__(self, Embed, params):
        self.auctionID = tf.compat.v1.placeholder(tf.int32, shape=(None), name = 'pid')
        self.current_user_id = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_user_id')
        self.current_user_history = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='current_user_history')
        self.current_user_bidtime = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_user_bidtime')
        self.current_user_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_user_neg_id')
       
        self.short_term_before_user_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_user_id')
        self.short_term_before_user_history = tf.compat.v1.placeholder(tf.float32, shape=[None,None,None], name='short_term_before_user_history')
        self.short_term_before_user_bidtime = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_user_bidtime')
        self.short_term_before_user_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_user_len')

        self.test_user_id = tf.compat.v1.placeholder(tf.float32, shape=(None), name='test_user_id')
        self.test_user_prob = tf.compat.v1.placeholder(tf.float32, shape=(None), name='test_user_probability')

        self.num_units = params.num_units

        # emb
        self.item_ID_emb = Embed.GetItemEmbedding(self.auctionID)   # 64 60

        self.transformer_INPUT_k_v = tf.tile(tf.expand_dims(self.item_ID_emb, axis=1),[1,params.short_term_size,1])

        # current user -> [uid, rate,bid] -> one-hot + 2 -> 非线性方程
        # 顺便处理neg
        self.pos_user_emb = tf.reshape(Embed.GetUserEmbedding(self.current_user_id), [-1,params.embed_size])    # 64 120
        self.neg_user_emb = tf.reshape(Embed.GetUserEmbedding(self.current_user_neg_id)  , [-1,5,params.embed_size])  # 64 99 120
        self.test_useremb = Embed.GetUserEmbedding(self.test_user_id)   # 64 100 60
        self.test_useremb = tf.reshape(self.test_useremb, [-1,100,params.embed_size])    # 64 100 120

        self.pos_user_emb=tf.compat.v1.layers.dense(inputs=self.pos_user_emb, units=params.embed_size,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.neg_user_emb=tf.compat.v1.layers.dense(inputs=self.neg_user_emb, units=params.embed_size,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.test_useremb=tf.compat.v1.layers.dense(inputs=self.test_useremb, units=params.embed_size,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        # 激活函数：tf.nn.relu

        #### MF for co-emb
        self.MFitems = tf.tile(tf.expand_dims(self.item_ID_emb, axis=1),[1,100,1])  # 64 100 120
        # self.MFusers = tf.tile(tf.expand_dims(self.pos_user_emb, axis=1),[1,1,1])   # 64 1 120
        # self.MFusers = tf.concat([self.MFusers, self.neg_user_emb],1)   # 不對啊，true一直在第一個？64 100 120
        
        self.MFusers = self.test_useremb        
        # 将user 和item的矩阵做点乘
        self.InferInputMF=tf.multiply(self.MFusers, self.MFitems)   # 64 100 120
        # 求和得出每对{user,item}的概率
        infer=tf.reduce_sum(self.InferInputMF, 2, name="inference") # 应该为64 100, prob

        self.MFoutput = infer

        # MF的损失函数:
        regularizer = tf.add(UW*tf.nn.l2_loss(self.MFusers), IW*tf.nn.l2_loss(self.MFitems), name="regularizer")
        global_step = tf.compat.v1.train.get_global_step()
        self.prob_batch = self.test_user_prob  # 64 100
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, self.prob_batch))# 64 100 with 64 100
        cost = tf.add(cost_l2, regularizer)
        self.opt_loss = cost
        self.train_op = tf.compat.v1.train.AdamOptimizer(params.learning_rate).minimize(cost, global_step=global_step)
        
        #### MF for co-emb



        # 3373
        # Define Query-Based Attention LSTM for User's short-term inference
        # 这里的embedding层处理得有点乱
        self.short_term_user_emb_id = Embed.GetUserEmbedding(self.short_term_before_user_id)   # 64 10 60
        self.short_term_user_emb_input = tf.concat([self.short_term_user_emb_id, self.short_term_before_user_history],-1)
        self.short_term_user_emb_input = tf.reshape(self.short_term_user_emb_input, [-1,params.short_term_size,3835])    # 64 100 120
        
        # MultiRec把第一层(dim+25)的输出作为正则化，第二层的输出用作相乘？(dim)
        self.short_term_user_emb=tf.compat.v1.layers.dense(inputs=self.short_term_user_emb_input, units=params.embed_size,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        # LSTM
        self.short_term_lstm_layer = tf.compat.v1.nn.rnn_cell.LSTMCell(self.num_units, forget_bias=1)
        self.short_term_lstm_outputs,_ = tf.compat.v1.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_user_emb, dtype="float32") 
        # output = 64 10 120

        # 求attention
        self.queryemb = self.item_ID_emb  # 64 60
        short_term_expand_q = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1,params.short_term_size,1])
        self.short_term_expand_q = short_term_expand_q
        self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))
        
        # attention * LSTM output
        self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1)
        # 链接item emb
        short_term_combine_user_item_emb =  tf.concat([self.item_ID_emb, self.user_short_term_emb], 1)
        
        self.short_term_combine_weights=tf.Variable(tf.random.uniform([2 * params.embed_size, params.embed_size]))
        self.short_term_combine_bias=tf.Variable(tf.random.uniform([params.embed_size]))

        self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)

        # Transformer 一些默认设定：
        # model_dim = params.embed_size
        # max_len = MAX_LEN
        # n_layer = N_LAYER
        # n_head = N_HEAD
        # drop_rate = DROP_RATE
        # self.embed = PositionEmbedding(max_len, model_dim)
        # self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        
        # x_embed = self.embed(self.transformer_INPUT_k_v)# x = 64,5,128 , y=1 5 128 和64 128不兼容 T
        # pad_mask = self._pad_mask(self.short_term_before_user_id)    # 64,1,1,5
        # self.mask = pad_mask# 64 1 1 5 120???
        # self.encoded_z = self.encoder.call(x_embed, x_embed, training=False, mask=pad_mask ) #64,5,128 
        # self.transformer_output = self.encoded_z[:, -1,:]
     

        # x = self.short_term_before_product_id
        # # tensorflow call函数：
        # # short_term_td_outputs( short_term_input_productemb + timedecay)
        # short_term_expand_q = tf.tile(tf.expand_dims(current_pid_time_2d, axis=1),[1,params.short_term_size,1])

        
        # # lstm 输出是： self.short_term_lstm_outputs
        # # transformer 输出是： self.encoded_z[:,-1,:] 最后一层
        # self.transformer_output = tf.tile(tf.expand_dims(self.encoded_z[:,-1,:], axis=1),[1,params.short_term_size,1])
        # self.short_term_attention = tf.nn.softmax(self.short_term_lstm_outputs)
        # self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))
        
        # self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1)
        # short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)


        # # loss func还有另外两个！！！useremb+query -> +time [ 64 128]

        self.useremb = self.short_term_useremb
        if params.loss_f == "Inner_product":
            self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
        elif params.loss_f == "MetricLearning":
            self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
        else:
            self.opt_loss, self.pos_loss, self.neg_loss = Pairwise_Loss_F(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
    
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.opt_loss = self.opt_loss + sum(reg_losses)

        # self.dis_pos = tf.reduce_sum(tf.multiply(self.useremb, self.pos_user_emb), axis=1) 
        # self.loss_pos = tf.reduce_mean(tf.math.log(tf.sigmoid(self.dis_pos)))

        # Optimiser
        step = tf.Variable(0, trainable=False)
        self.opt = gradients(
            opt=tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.compat.v1.trainable_variables(),
            step=step
        )

    # ((uid, cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_flag, cur_before_pids_attr,current_pid_pos,cur_pids_attr))
    def step(self, session, pid, cuid, cuhis, cbidtime, sbu, sbuhis, sbidtime, sbul, cnuid,tid,tprob, testmode = False):
        # cu_train,cuhis_train,cuBT_train,sbu_train,sbuhis_train,
        input_feed = {}
        input_feed[self.auctionID.name] = pid   # 

        input_feed[self.current_user_id.name] = cuid    # 
        input_feed[self.current_user_history.name] = cuhis    #  64，97【变】
        input_feed[self.current_user_bidtime.name] = cbidtime    # 
        input_feed[self.current_user_neg_id.name] = cnuid # 64 5 

        input_feed[self.short_term_before_user_id.name] = sbu# 64 10 3
        input_feed[self.short_term_before_user_history.name] = sbuhis # 64 10 97【变】
        input_feed[self.short_term_before_user_bidtime.name] = sbidtime# 64 10
        input_feed[self.short_term_before_user_len.name] = sbul # 64

        input_feed[self.test_user_id.name] = tid
        input_feed[self.test_user_prob.name] = tprob

        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
            # output_feed = [self.train_op, self.opt_loss] # MF 的error
            # output_feed = [ self.opt_loss,self.opt_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb
            output_feed = [tf.shape(self.auctionID)[0], u_plus_q, self.test_useremb,self.opt_loss]

            # MF 的error
            # output_feed = [tf.shape(self.auctionID)[0], u_plus_q, self.opt_loss]
            
        outputs = session.run(output_feed, input_feed)

        # short_feed = [ self.useremb, self.pos_user_emb,self.pos_loss, self.neg_loss]
        # short = session.run(short_feed, input_feed)
        # for i in range(len(short)):
        #     print("__________________________________")
        #     print(short[i])
        #     print(short[i].shape)

        return outputs
        # return short

    def _pad_bool(self, seqs):
        padding_idx=0
        return tf.math.equal(seqs, padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self, seqs):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        # print( self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :], mask[tf.newaxis, tf.newaxis, :, :])
        mask = tf.where( self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :],1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)



