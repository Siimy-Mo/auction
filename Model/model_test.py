import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import numpy as np
from tensorflow.contrib import rnn
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
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
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def Loss_F(useremb, queryemb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb + queryemb
    dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis = 1), -1.0)
    log_u_plus_q_minus_i_pos = tf.math.log(tf.sigmoid(dis_pos))
    
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    log_u_plus_q_minus_i_neg = tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg)),axis=1)
    batch_loss = -1 * (log_u_plus_q_minus_i_neg + log_u_plus_q_minus_i_pos)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(-dis_pos), tf.reduce_mean(dis_neg)

def Pairwise_Loss_F(useremb, queryemb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb + queryemb
    #dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis=1), tf.cast(k, tf.float32))
    dis_pos = tf.norm(u_plus_q - product_pos, ord=2, axis=1)
    dis_pos = tf.reshape(dis_pos, [-1, 1])
    #dis_pos = tf.tile(dis_pos, [1, k])

    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1, k, 1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    
    batch_loss = tf.multiply(tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg - k * dis_pos) + 1e-6), axis=1), -1.0)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)

def Inner_product(useremb, queryemb, product_pos, product_neg, k):
    # u_plus_q = user+query
    u_plus_q = useremb + queryemb
    
    #uq=u_plus_q.unsqueeze(2)  # skip
    #itp = item_pos.unsqueeze(1) # skip
    #pos_skip = torch.bmm(itp, uq) # skip
    #transpose_product_pos = tf.transpose(product_pos)

    #dis_pos = tf.matmul(u_plus_q, product_pos, transpose_b=True)
    dis_pos = tf.reduce_sum(tf.multiply(u_plus_q, product_pos), axis=1) 
    #dis_pos = -1.0 * tf.matmul(u_plus_q, product_pos, transpose_b=True)
    #dis_pos = tf.multiply(u_plus_q, product_pos)
    #loss_pos = pos_skip.sigmoid().log().mean()
    loss_pos = tf.reduce_mean(tf.math.log(tf.sigmoid(dis_pos)))
    #dis_pos = tf.reshape(dis_pos, [-1, 1])
    
    #itn = items_neg.unsqueeze(2) # skip
    #batch_size, neg_num, em_dim = items_neg.shape
        #neg_skip = torch.empty(batch_size, neg_num, 1)
    #for i in range(self.batch_size):
        #neg_skip[i] = torch.matmul(itn[i],uq[i]).squeeze(2)
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    #transpose_product_neg = tf.transpose(product_neg,perm=[0, 2, 1])
    
    #dis_neg = tf.matmul(expand_u_plus_q, transpose_product_neg)
    dis_neg = tf.reduce_sum(tf.multiply(expand_u_plus_q, product_neg), axis=2) 
    #dis_neg = tf.multiply(expand_u_plus_q, product_neg)
    # loss_neg = neg_skip.mul(-1.0).sigmoid().log().sum(dim=1).mean()
    loss_neg = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.sigmoid(tf.multiply(dis_neg, -1.0))),axis=1))
    #loss_neg = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg)),axis=1))
    #dis_neg = tf.reshape(dis_neg, [-1, 1])
    batch_loss = -1.0*(loss_pos + loss_neg)
    
    #return tf.reduce_mean(batch_loss), tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)
    return batch_loss, tf.reduce_mean(dis_pos), tf.reduce_mean(dis_neg)
    
    #loss_neg = neg_skip.mul(-1.0).sigmoid().log().sum(dim=1).mean()
    #batch_loss = -1.0*(loss_pos + loss_neg)
    #return batch_loss, pos_skip.mean(), neg_skip.mean()

class Seq(object):
    def __init__(self, Embed, params):
        self.UserID = tf.compat.v1.placeholder(tf.int32, shape=(None), name = 'uid')
        self.current_product_id = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_id')
        self.current_product_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_product_neg_id')
        self.short_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='short_term_query_id')
        self.short_term_query_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_query_len')
        self.short_term_query_len_mask = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='short_term_query_len_mask')
       
        self.short_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_id')
        self.short_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_len')
        self.short_term_before_product_flag = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_flag')

        self.durations = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_flag1')
        self.openbids = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_flag2')


        self.num_units = params.num_units
        self.long_term_size = params.long_term_size

        #batch_size = tf.shape(self.UserID)[0]
        self.productemb = Embed.GetAllProductEmbedding()
        # emb
        self.user_ID_emb = Embed.GetUserEmbedding(self.UserID)

        # current query emb
        self.queryemb = Embed.GetProductEmbedding(self.current_product_id)
        # self.queryemb = Embed.GetAuctionEmbedding(self.current_product_id)

        self.beforeproductemb = Embed.GetAuctionEmbedding(self.short_term_before_product_id)

        # current query emb
        # self.queryemb = Embed.GetQueryEmbedding(self.short_term_query_id, self.short_term_query_len, self.short_term_query_len_mask)
        
        # positive emb

        self.product_pos_emb = Embed.GetProductEmbedding(self.current_product_id)
        self.product_neg_emb = Embed.GetProductEmbedding(self.current_product_neg_id)
        
        self.duration_emb, self.openbid_emb = Embed.GetAuctionEmbedding_2(self.durations, self.openbids)
        # print(self.duration_emb.shape, self.openbid_emb.shape)
        self.short_term_before_productemb = tf.concat([self.duration_emb, self.openbid_emb],0)
        # print(self.short_term_before_productemb.shape)
        self.short_term_before_productemb = Embed.GetProductEmbedding(self.short_term_before_product_id)
        

        # Define Query-Based Attention LSTM for User's short-term inference
        # 設置LSTM層數，並輸入emb然後得到hidden embedding
        self.short_term_lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_before_productemb, dtype="float32")  # hidden: latent representation []

        self.short_term_lstm_outputs = tf.reduce_sum(self.short_term_lstm_outputs,axis=1)
        # 重要↓ # short_term_expand_q = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1,params.short_term_size,1])   # deal with query representation
        short_term_expand_q = tf.tile(tf.expand_dims(self.productemb, axis=1),[1,params.short_term_size,1])   # deal with query representation. queryemb -> product
        
        self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))   # short-term attentions []， multiply is score function
        
        self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1) # 壓縮求和，降維
        short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)  # short-term preference公式所需

        self.short_term_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        self.short_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))


        # self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)  # short-term preference
        self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)  # short-term preference
        print(self.short_term_useremb.shape)

        if params.user_emb == "Short_term":
        #     self.useremb = self.long_term_useremb * self.long_short_rate + (1 - self.long_short_rate) * self.short_term_useremb
        # elif params.user_emb == "Complete":
            self.useremb = self.short_term_useremb
        # elif params.user_emb == "Long_term":
        #     self.useremb = self.long_term_useremb
        else:
            self.useremb = self.user_ID_emb

        # self.long_short_rate = tf.sigmoid(self.long_term_before_product_len - params.short_term_size)
        # 
        #self.useremb = self.short_term_useremb
        self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.productemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)

        if params.loss_f == "Inner_product":
            self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.productemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        elif params.loss_f == "MetricLearning":
            self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.productemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        else:
            self.opt_loss, self.pos_loss, self.neg_loss = Pairwise_Loss_F(self.useremb, self.productemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
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
    def step(self, session, uid, sbpid, sbpidlen, sbflag, cpid, cpnid, testmode = False):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.short_term_before_product_id.name] = sbpid
        input_feed[self.short_term_before_product_len.name] = sbpidlen
        input_feed[self.short_term_before_product_flag.name] = sbflag
        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid
        # input_feed[self.durations.name] = duration
        # input_feed[self.openbids.name] =openbid
        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb + self.queryemb
            output_feed = [tf.shape(self.UserID)[0], u_plus_q, self.productemb]
        print('__________________________________________')
        print(self.opt, self.opt_loss, self.pos_loss, self.neg_loss)
        outputs = session.run(output_feed, input_feed)
        return outputs

    def summary(self, session, summarys, uid, cpid, cpnid, qp, lp, qlm, HR, hr, MRR, mrr, NDCG, ndcg,AVG_LOSS,avg_loss):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid
        input_feed[self.query_id.name] = qp
        input_feed[self.query_len.name] = lp
        input_feed[self.query_len_mask.name] = qlm
        input_feed[HR.name] = hr
        input_feed[MRR.name] = mrr
        input_feed[NDCG.name] = ndcg
        input_feed[AVG_LOSS.name] = avg_loss
        output_feed = [summarys]
        outputs = session.run(output_feed, input_feed)
        return outputs