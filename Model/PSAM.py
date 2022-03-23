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
    log_u_plus_q_minus_i_neg = tf.reduce_sum(tf.log(tf.sigmoid(dis_neg)),axis=1)
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
    
    batch_loss = tf.multiply(tf.reduce_sum(tf.log(tf.sigmoid(dis_neg - k * dis_pos) + 1e-6), axis=1), -1.0)
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

class PSAM(object):
    def __init__(self, Embed, params):
        self.UserID = tf.compat.v1.placeholder(tf.int32, shape=(None), name = 'uid')
        self.current_product_id = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_id')
        self.current_product_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_product_neg_id')
        self.short_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='short_term_query_id')
        self.short_term_query_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_query_len')
        self.short_term_query_len_mask = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='short_term_query_len_mask')
        self.long_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name='long_term_query_id')
        self.long_term_query_len =  tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='long_term_query_len')
        self.long_term_query_len_mask =  tf.compat.v1.placeholder(tf.float32, shape=[None, None, None], name='long_term_query_len_mask')
        
        self.short_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_id')
        self.short_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_len')
        self.long_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='long_term_before_product_id')
        self.long_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='long_term_before_product_len')
        self.num_units = params.num_units
        self.long_term_size = params.long_term_size
        
        #batch_size = tf.shape(self.UserID)[0]
        self.productemb = Embed.GetAllProductEmbedding()
        # emb
        self.user_ID_emb = Embed.GetUserEmbedding(self.UserID)

        # current query emb
        self.queryemb = Embed.GetQueryEmbedding(self.short_term_query_id, self.short_term_query_len, self.short_term_query_len_mask)

        self.beforequeryemb = Embed.GetQueryEmbedding(self.long_term_query_id, self.long_term_query_len, self.long_term_query_len_mask)
        
        # positive emb
        self.product_pos_emb = Embed.GetProductEmbedding(self.current_product_id)
        self.product_neg_emb = Embed.GetProductEmbedding(self.current_product_neg_id)
        
        self.short_term_before_productemb = Embed.GetProductEmbedding(self.short_term_before_product_id)
        self.long_term_before_productemb = Embed.GetProductEmbedding(self.long_term_before_product_id)
        
        # short term
        self.short_term_lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_before_productemb, dtype="float32")
        short_term_expand_q = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1,params.short_term_size,1])
        self.short_term_attention = tf.nn.softmax(tf.multiply(self.short_term_lstm_outputs, short_term_expand_q))
        
        self.user_short_term_emb = tf.reduce_sum(tf.multiply(self.short_term_attention, self.short_term_lstm_outputs),axis=1)
        short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)
        
        self.short_term_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        self.short_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        
        
        self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)
        #self.useremb = tf.layers.dense(inputs=combine_user_item_emb, units=params.embed_size, activation=None)
        

        # # Define Memory Network for User's long-term inference
        # long_term_expand_user_emb = tf.tile(tf.expand_dims(self.user_ID_emb, axis=1),[1, self.long_term_size, 1])
        # long_term_expand_query_emb = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1, self.long_term_size, 1])
        # long_term_scores = tf.multiply(long_term_expand_user_emb, long_term_expand_query_emb) +  tf.multiply(self.long_term_before_productemb, long_term_expand_query_emb)
        
        # long_term_memory_mask = tf.tile(tf.expand_dims(tf.cast(tf.not_equal(self.long_term_before_product_id,0), dtype=tf.float32),axis=-1),[1,1,params.embed_size])
        # self.long_term_scores = tf.reduce_sum(tf.multiply(long_term_scores, long_term_memory_mask),axis=-1)
        
        
        # self.long_term_attention = tf.nn.softmax(self.long_term_scores, name='LongTermAttention')

        # # [batch_Size, max_pro_length] => [batch_Size, 1, max_pro_length]
        # probs_temp = tf.expand_dims(self.long_term_attention, 1, name='TransformLongTermAttention')

        # #  output_memory: [batch size, max_pro_length, emb_size]
        # #  Transpose: [batch_Size, emb_size, max_pro_length]
        # c_temp = tf.transpose(self.beforequeryemb, [0, 2, 1], name='TransformOutputMemory')

        # # Apply a weighted scalar or attention to the external memory
        # # [batch size, 1, <max length>] * [batch size, embedding size, <max length>]
        # neighborhood = tf.multiply(c_temp, probs_temp, name='WeightedNeighborhood')
        # # Sum the weighted memories together
        # # Input:  [batch Size, embedding size, <max length>]
        # # Output: [Batch Size, Embedding Size]
        # # Weighted output vector
        # self.long_term_useremb = tf.reduce_sum(neighborhood, axis=2, name='OutputNeighborhood')

        # second
        
        # query-Attention
        # long_term_expand_query_emb:[batch_size, long_term_size, emb_size]
        long_term_expand_query_emb = tf.tile(tf.expand_dims(self.queryemb, axis=1),[1, self.long_term_size, 1])
        normalize_beforequeryemb = tf.nn.l2_normalize(self.beforequeryemb, axis=2)
        normalize_currentqueryemb = tf.nn.l2_normalize(long_term_expand_query_emb, axis=2)
        
        current_query_similarity = tf.reduce_sum(tf.multiply(normalize_beforequeryemb, normalize_currentqueryemb), axis=2)
        long_term_memory_mask = tf.cast(tf.not_equal(self.long_term_before_product_id,0), dtype=tf.float32)
        # current_query_similarity -> [batch_size, long_term_size]
        current_query_similarity = tf.multiply(current_query_similarity, long_term_memory_mask)

        # current_query_similarity -> [batch_size]
        l1_normalize_current_query_similarity = tf.reduce_sum(current_query_similarity, axis=1)
        expand_l1_normalize_current_query_similarity = tf.tile(tf.expand_dims(l1_normalize_current_query_similarity, axis=-1), [1, params.long_term_size])

        self.long_term_attention = tf.divide(current_query_similarity, expand_l1_normalize_current_query_similarity)
        expand_long_term_attention = tf.tile(tf.expand_dims(self.long_term_attention, axis=-1), [1, 1, params.embed_size])
        
        # long_term_query_product-> [batch_size, emb_size]
        self.long_term_query_product = tf.reduce_sum(tf.multiply(expand_long_term_attention, self.long_term_before_productemb), axis=1)
        
        long_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.long_term_query_product], 1)
        
        self.long_term_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        self.long_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        self.long_term_useremb = tf.tanh(tf.matmul(long_term_combine_user_item_emb, self.long_term_combine_weights) + self.long_term_combine_bias)

        # combine_user_emb =  tf.concat([self.long_term_useremb, self.short_term_useremb], 1)
        # self.user_emb_combine_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        # self.user_emb_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))
        # self.useremb = tf.tanh(tf.matmul(combine_user_emb, self.user_emb_combine_weights) + self.user_emb_combine_bias)

        # hyper parameter
        user_long_emb_weights= tf.Variable(tf.random_normal([params.embed_size, params.embed_size]))
        user_short_emb_weights = tf.Variable(tf.random_normal([params.embed_size, params.embed_size]))
        user_long_short_emb_bias = tf.Variable(tf.random_normal([params.embed_size]))
        self.long_short_rate = tf.sigmoid(tf.matmul(self.long_term_useremb, user_long_emb_weights) + tf.matmul(self.short_term_useremb, user_short_emb_weights) + user_long_short_emb_bias)

        if params.user_emb == "Complete":
            self.useremb = self.long_term_useremb * self.long_short_rate + (1 - self.long_short_rate) * self.short_term_useremb
        elif params.user_emb == "Short_term":
            self.useremb = self.short_term_useremb
        elif params.user_emb == "Long_term":
            self.useremb = self.long_term_useremb
        else:
            self.useremb = self.user_ID_emb

        # self.long_short_rate = tf.sigmoid(self.long_term_before_product_len - params.short_term_size)
        # 
        #self.useremb = self.short_term_useremb

        if params.loss_f == "Inner_product":
            self.opt_loss, self.pos_loss, self.neg_loss = Inner_product(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        elif params.loss_f == "MetricLearning":
            self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        else:
            self.opt_loss, self.pos_loss, self.neg_loss = Pairwise_Loss_F(self.useremb, self.queryemb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.opt_loss = self.opt_loss + sum(reg_losses)
        
        
        
        # Optimiser
        step = tf.Variable(0, trainable=False)
        
        self.opt = gradients(
            opt=tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.compat.v1.trainable_variables(),
            step=step
        )
    # (session, u_train, bpp_train, sl_train, lbpp_train, lsl_train, cpp_train, cni_train, qp_train, lp_train, bqp_train, blp_train, query_len_mask_train, long_query_len_mask_train)
    def step(self, session, uid, sbpid, sbpidlen, lbpid, lbpidlen, cpid, cpnid, qp, lp, bqp, blp,qlm, bqlm, testmode = False):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.short_term_before_product_id.name] = sbpid
        input_feed[self.short_term_before_product_len.name] = sbpidlen
        input_feed[self.long_term_before_product_id.name] = lbpid
        input_feed[self.long_term_before_product_len.name] =  lbpidlen
        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid
        input_feed[self.short_term_query_id.name] = qp
        input_feed[self.short_term_query_len.name] = lp
        input_feed[self.short_term_query_len_mask.name] = qlm
        input_feed[self.long_term_query_id.name] = bqp
        input_feed[self.long_term_query_len.name] = blp
        input_feed[self.long_term_query_len_mask.name] = bqlm
        
        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb + self.queryemb
            output_feed = [tf.shape(self.UserID)[0], u_plus_q, self.productemb]
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

def init_dataset(self, UserData, short_term_size, long_term_size, weights=True):
        try:
            self.data_X = []
            for U in range(len(UserData)):
                uid = self.user_2_id[UserData[U].UserID]
                # 每个User的数据，方便以后划分数据集
                User_Data_X = []
                
                UserItemLen = len(UserData[U].UserPurchaseList)
                
                # review
                before_review_text_list = []
                before_review_text_len_list = []
                
                # purchase list
                before_pids_pos = []
                
                # the query of purchase list
                before_querylist_pos = []
                before_querylist_pos_len = []
                
                # 向前补充short_term_size-1个信息
                for k in range(0, short_term_size - 1):
                    before_pids_pos.append(self.product_2_id['<pad>'])
                    padding_review_text = ""
                    before_text_ids, before_text_len = self.trans_to_ids(padding_review_text, self.max_review_len)
                    before_review_text_list.append(before_text_ids)
                    before_review_text_len_list.append(before_text_len)
                
                
                for l in range(1, UserItemLen):
                    # v(i-short_term_size), v(i - windows_size +1),...,v(i-1)  
                    before_pid_pos = self.product_2_id[UserData[U].UserPurchaseList[l-1]['asin']]
                    before_pids_pos.append(before_pid_pos)
                    before_review_text = UserData[U].UserPurchaseList[l-1]['reviewText']
                    try:
                        before_text_ids, before_text_len = self.trans_to_ids(before_review_text, self.max_review_len)
                    except:
                        before_text_len = 0
                        before_text_ids = []
                        for i in range(self.max_review_len):
                            before_text_ids.append(self.word_2_id['<pad>'])
                    before_review_text_list.append(before_text_ids)
                    before_review_text_len_list.append(before_text_len)
                    try:
                        before_query_text_array_pos = self.product_2_query[self.id_2_product[before_pid_pos]]
                        before_Qids_pos, before_Len_pos = self.trans_to_ids(Q_text_array_pos[0], self.max_query_len)
                        before_querylist_pos.append(before_Qids_pos)
                        before_querylist_pos_len.append(before_Len_pos)
                    except:
                        print("User:%s Item:%s has not query!\n" % (str(UserData[U].UserID), str(UserData[U].UserPurchaseList[l-1]['asin'])))
                        null_query_list = []
                        for i in range(0, self.max_query_len):
                            null_query_list.append(self.word_2_id['<pad>'])
                        before_querylist_pos.append(null_query_list)
                        before_querylist_pos_len.append(0)
                    # vi
                    current_pid_pos = self.product_2_id[UserData[U].UserPurchaseList[l]['asin']]
                    cur_before_pids_pos = before_pids_pos[-short_term_size:]

                    # generate long-term sequence
                    if len(before_pids_pos) < long_term_size:
                        cur_long_before_pids_pos = [self.product_2_id['<pad>']] * (long_term_size - len(before_pids_pos)) + before_pids_pos
                        cur_long_before_pids_pos_len = len(before_pids_pos) - (short_term_size - 1)
                    else:
                        cur_long_before_pids_pos = before_pids_pos[-long_term_size:]
                        cur_long_before_pids_pos_len = len(cur_long_before_pids_pos)
                    

                    if l < short_term_size:
                        cur_before_pids_pos_len = l
                    else:
                        cur_before_pids_pos_len = short_term_size
                    self.before_pid_pos[str(uid) + "_" + str(current_pid_pos)] = cur_before_pids_pos
                    self.before_textlist[str(uid) + "_" + str(current_pid_pos)] = before_review_text_list[-short_term_size:]
                    self.before_textlenlist[str(uid) + "_" + str(current_pid_pos)] = before_review_text_len_list[-short_term_size:]
                    self.before_querylist_pos[str(uid) + "_" + str(current_pid_pos)] = before_querylist_pos[-short_term_size:]
                    self.nes_weight[current_pid_pos] += 1
                    current_text =  UserData[U].UserPurchaseList[l]['reviewText']
                    try:
                        current_text_ids, current_text_Len = self.trans_to_ids(current_text, self.max_review_len)
                    except:
                        current_text_Len = 0
                        current_text_ids = []
                        for i in range(self.max_review_len):
                            current_text_ids.append(self.word_2_id['<pad>'])
                    try:
                        Q_text_array_pos = self.product_2_query[self.id_2_product[current_pid_pos]]
                    except:
                        # vi物品没有query，不加入数据集
                        Q_text_array_pos = []
                    for Qi in range(len(Q_text_array_pos)):
                        try:
                            Qids_pos, Len_pos = self.trans_to_ids(Q_text_array_pos[Qi], self.max_query_len)
                        except:
                            break
                        product_len_mask = [0.] * (short_term_size - cur_before_pids_pos_len) + [1.] * cur_before_pids_pos_len
                        query_len_mask = [1.] * Len_pos + [0.] * (self.max_query_len - Len_pos)


                        null_query_list = [self.word_2_id['<pad>']] * self.max_query_len
                        if len(before_querylist_pos) < self.long_term_size:
                            cur_long_before_query_pos =  [null_query_list] * (self.long_term_size - len(before_querylist_pos)) + before_querylist_pos
                            null_query_mask = [0.] * self.max_query_len
                            cur_long_before_query_len_mask = [null_query_mask] * (self.long_term_size - len(before_querylist_pos_len)) + [[1.] * i + [0.] * (self.max_query_len - i) for i in before_querylist_pos_len]
                            cur_long_before_query_pos_len = [0] * (self.long_term_size - len(before_querylist_pos_len)) + before_querylist_pos_len
                        else:
                            cur_long_before_query_pos = before_querylist_pos[-self.long_term_size:]
                            cur_long_before_query_len_mask = [[1.] * i + [0.] * (self.max_query_len - i) for i in before_querylist_pos_len[-self.long_term_size:]]
                            cur_long_before_query_pos_len = before_querylist_pos_len[-self.long_term_size:]
                            
                        User_Data_X.append((uid, cur_before_pids_pos, cur_before_pids_pos_len, current_pid_pos, cur_long_before_pids_pos, cur_long_before_pids_pos_len, \
                                            Qids_pos, Len_pos, cur_long_before_query_pos, cur_long_before_query_pos_len,\
                                            current_text_ids, current_text_Len, product_len_mask, query_len_mask,cur_long_before_query_len_mask))
                        #User_Data_X.append((self.Tran_Uid_2_vid([uid]), self.Tran_Pid_2_vid(cur_before_pids_pos), self.Tran_Pid_2_vid([current_pid_pos]), self.Tran_Wid_2_vid(Qids_pos),
                                            #Len_pos, self.Tran_Wid_2_vid(current_text_ids), current_text_Len))
                        try:
                            self.userReviewsCount[uid] += 1
                            self.userReviewsCounter[uid] += 1
                        except:
                            self.userReviewsCount[uid] = 1
                            self.userReviewsCounter[uid] = 1    
                self.data_X.append(User_Data_X)
            
            '''
            数据集划分-根据用户进行划分，划分比例为7:2:1
            '''
            for u in self.data_X:
                u_len = len(u)
                u_train_data = u[:int(0.7*u_len)]
                u_validation_data = u[int(0.7*u_len):int(0.9*u_len)]
                u_test_data = u[int(0.9*u_len):]
                self.train_data.extend(u_train_data)
                self.test_data.extend(u_test_data)
                self.eval_data.extend(u_validation_data)
                
                
            if weights is not False:
                wf = np.power(self.nes_weight, 0.75)
                wf = wf / wf.sum()
                self.weights = wf
                wf = np.power(self.word_weight, 0.75)
                wf = wf / wf.sum()
                self.word_weight = wf
        
        except Exception as e:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')