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


MAX_LEN = 5
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1
MFSIZE = 60

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
    u_plus_q = useremb 
    dis_pos = tf.reduce_sum(tf.multiply(u_plus_q, product_pos), axis=1) 
    loss_pos = tf.reduce_mean(tf.math.log(tf.sigmoid(dis_pos)))
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
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
        self.current_user_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_user_neg_id')
       
        self.short_term_before_user_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_user_id')
        self.short_term_before_user_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_user_len')


        self.num_units = params.num_units

        # test emb: 生成64 100 256的对应的user emb矩阵返回：
        self.test_useremb = Embed.GetAllUserEmbedding()

        # emb
        self.item_ID_emb = Embed.GetItemEmbedding(self.auctionID)
        self.transformer_INPUT_k_v = tf.tile(tf.expand_dims(self.item_ID_emb, axis=1),[1,params.short_term_size,1])

        # current user -> [uid, rate,bid] -> one-hot + 2 -> 非线性方程
        # 顺便处理neg
        self.pos_user_emb = Embed.GetUserEmbedding(self.current_user_id)
        self.neg_user_emb = Embed.GetUserEmbedding(self.current_user_neg_id)

        # self.neg_user_emb = Embed.GetUserEmbedding(self.short_term_before_user_id)

        # 3373
        self.short_term_user_emb = Embed.GetUserEmbedding(self.short_term_before_user_id)
        # self.InferInputMF=tf.multiply(self.transformer_INPUT_k_v, self.transformer_INPUT_q)

        # # self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_td_outputs, dtype="float32")  # 加type属性
        # self.short_term_td_outputs = tf.nn.softmax(tf.multiply(self.beforeproductemb, self.timeDecay)) #考虑时间差的影响, 64 5 128

        # Define Query-Based Attention LSTM for User's short-term inference
        self.short_term_lstm_layer = tf.compat.v1.nn.rnn_cell.LSTMCell(self.num_units * 2, forget_bias=1)
        self.short_term_lstm_outputs,_ = tf.compat.v1.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_user_emb, dtype="float32") 
        # 64 5 256

        # Transformer 一些默认设定：
        model_dim = params.embed_size*2
        max_len = MAX_LEN
        n_layer = N_LAYER
        n_head = N_HEAD
        drop_rate = DROP_RATE
        self.embed = PositionEmbedding(max_len, model_dim)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        
        x_embed = self.embed(self.short_term_user_emb)# x = 64,5,128 , y=1 5 128 和64 128不兼容 T
        pad_mask = self._pad_mask(self.short_term_before_user_id)    # 64,1,1,5
        self.mask = pad_mask# 64 1 1 5 120???
        self.encoded_z = self.encoder.call(x_embed, x_embed, training=False, mask=pad_mask ) #64,5,128 
        self.transformer_output = self.encoded_z[:, -1,:]
        # # self.short_term_before_productemb = Embed.GetProductEmbedding(self.short_term_before_product_id)   #原来的productemb，64 5 128
        # self.long_term_before_productemb = Embed.GetProductEmbedding(self.long_term_before_product_id)   #原来的productemb，64 5 128
 

        # # # 处理时间衰减比重 (64,5)-> (64,5,128) 但是没有半衰系数
        # # self.timeDecay = tf.tile(tf.expand_dims(tf.exp(self.time_diff, name=None), axis=2),[1,1,params.embed_size])
        
        # # # self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_td_outputs, dtype="float32")  # 加type属性
        # # self.short_term_td_outputs = tf.nn.softmax(tf.multiply(self.beforeproductemb, self.timeDecay)) #考虑时间差的影响, 64 5 128

        # # Define Query-Based Attention LSTM for User's short-term inference
        # self.short_term_lstm_layer = tf.compat.v1.nn.rnn_cell.LSTMCell(self.num_units, forget_bias=1)
        # self.short_term_lstm_outputs,_ = tf.compat.v1.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_input_productemb, dtype="float32")  # hidden: latent representation [64,5,100]


        # x = self.short_term_before_product_id
        # # tensorflow call函数：
        # # short_term_td_outputs( short_term_input_productemb + timedecay)
        # short_term_expand_q = tf.tile(tf.expand_dims(current_pid_time_2d, axis=1),[1,params.short_term_size,1])

        # x_embed = self.embed(self.short_term_input_productemb)# x = 64,5,128 , y=1 5 128 和64 128不兼容 T
        # pad_mask = self._pad_mask(x)    # 64,1,1,5
        # self.encoded_z = self.encoder.call(x_embed, x_embed, training=False, mask=pad_mask ) #64,5,128 

        # # current_pid_bidderList
        
        # # lstm 输出是： self.short_term_lstm_outputs
        # # transformer 输出是： self.encoded_z[:,-1,:] 最后一层
        # self.transformer_output = tf.tile(tf.expand_dims(self.encoded_z[:,-1,:], axis=1),[1,params.short_term_size,1])
        # self.short_term_attention = tf.nn.softmax(self.short_term_lstm_outputs)
        # self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))
        
        # self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1)
        # short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)

        # # 全连接层。
        # self.short_term_combine_weights=tf.Variable(tf.random.normal([2 * params.embed_size, params.embed_size]))
        # self.short_term_combine_bias=tf.Variable(tf.random.normal([params.embed_size]))
        # self.long_term_combine_weights=tf.Variable(tf.random.normal([2 * params.embed_size, params.embed_size]))
        # self.long_term_combine_bias=tf.Variable(tf.random.normal([params.embed_size]))
        
        # self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)  # short-term preference


        # if params.user_emb == "Complete":
        #     self.useremb = self.long_term_useremb * self.long_short_rate + (1 - self.long_short_rate) * self.short_term_useremb
        # elif params.user_emb == "Short_term":
        # # if params.user_emb == "Short_term":

        self.user_short_term_emb = tf.reduce_sum(self.short_term_lstm_outputs,axis=1)
        # self.short_term_combine_weights=tf.Variable(tf.random.normal([2 * params.embed_size, params.embed_size]))
        # self.short_term_combine_bias=tf.Variable(tf.random.normal([params.embed_size]))
        # self.short_term_useremb = tf.tanh(tf.matmul(self.user_short_term_emb, self.short_term_combine_weights) + self.short_term_combine_bias)  # short-term preference

        
        self.useremb = self.transformer_output
        #     # self.useremb = self.user_short_term_emb #without user emb
        # elif params.user_emb == "Long_term":
        #     self.useremb = self.long_term_useremb
        #     # self.useremb = self.encoded_z[:,-1,:]  # last layer of transformer encoder
        # else:
        #     self.useremb = self.user_ID_emb
        # # ------------------------------ MLP MODEL
        # # 输入节点数
        # # in_units = 784
        # # # 隐含层节点数
        # # h1_units = 300
        # # # variable存储模型参数
        # # # Variable长期存在并且每轮更新
        # # # 隐含层初始化为截断的正态分布，标准差为0.1
        # # w1 = tf.Variable(tf.random.truncated_normal([params.embed_size,h1_units],stddev = 0.1))
        # # # 隐含层偏置用0初始化
        # # b1 = tf.Variable(tf.zeros([h1_units]))
        # # # 输出层用0初始化，且维数为10
        # # w2 = tf.Variable(tf.zeros([h1_units,params.embed_size]))
        # # b2 = tf.Variable(tf.zeros([params.embed_size]))

        # # x = self.short_term_input_productemb # [64,5,128]
        # # x = tf.reduce_sum(x,axis=1) # [64,128]
        # # # 隐含层结构
        # # # relu激活函数是梯度弥散的trick
        # # hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
        # # # 对隐含层输出进行dropout
        # # hidden1_drop = keras.layers.Dropout( rate = 0.1)
        # # hidden1_drop_output = hidden1_drop(hidden1,False)
        # # # 输出层结构
        # # self.useremb = tf.nn.softmax(tf.matmul(hidden1_drop_output,w2)+b2)


        # # loss func还有另外两个！！！useremb+query -> +time [ 64 128]

        if params.loss_f == "Inner_product":
            self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
        elif params.loss_f == "MetricLearning":
            self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
        else:
            self.opt_loss, self.pos_loss, self.neg_loss = Pairwise_Loss_F(self.useremb, self.pos_user_emb, self.neg_user_emb, params.neg_sample_num)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.opt_loss = self.opt_loss + sum(reg_losses)
        
        
        
        # Optimiser
        step = tf.Variable(0, trainable=False)
        
        self.opt = gradients(
            opt=tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.compat.v1.trainable_variables(),
            step=step
        )

    # ((uid, cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_flag, cur_before_pids_attr,current_pid_pos,cur_pids_attr))
    def step(self, session, pid, cuid, sbu, sbul, cnuid, testmode = False):
        # p,p_attrs_test,cu_test,sbu_test,sbul_test,cni_test
        input_feed = {}
        input_feed[self.auctionID.name] = pid   # 64 606

        input_feed[self.current_user_id.name] = cuid    # 64 3373
        input_feed[self.current_user_neg_id.name] = cnuid # 64 5 3373

        input_feed[self.short_term_before_user_id.name] = sbu# 64 5 3
        input_feed[self.short_term_before_user_len.name] = sbul # 64

        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb
            output_feed = [tf.shape(self.auctionID)[0], u_plus_q, self.test_useremb,self.opt_loss]
            
        outputs = session.run(output_feed, input_feed)

        short_feed = [self.item_ID_emb ]
        short = session.run(short_feed, input_feed)
        for i in range(len(short)):
            print("__________________________________")
            print(short[i])
            print(short[i].shape)

        return outputs
        # return short

    def summary(self, session, summarys):
        # input_feed = {}
        # input_feed[self.UserID.name] = uid
        # input_feed[self.current_product_id.name] = cpid
        # input_feed[self.current_product_neg_id.name] = cpnid
        # input_feed[HR.name] = hr
        # input_feed[MRR.name] = mrr
        # input_feed[NDCG.name] = ndcg
        # input_feed[AVG_LOSS.name] = avg_loss
        output_feed = [summarys]
        # outputs = session.run(output_feed, input_feed)
        outputs = session.run(output_feed)
        return outputs

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



