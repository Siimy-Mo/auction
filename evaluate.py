import numpy as np
import tensorflow as tf
import random
class Evaluate(object):
    def __init__(self, params):

        self.currentid = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_id')
        self.itememb = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='item_emb')
        self.allproductemb = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='allproduct_emb')
        item_expand_dim_num = tf.shape(self.allproductemb)[0]
        self.expand_item_emb = tf.tile(tf.expand_dims(self.itememb, axis=1),[1,item_expand_dim_num,1])

        allproduct_expand_dim_num = tf.shape(self.itememb)[0]
        self.expand_allproduct_num = tf.tile(tf.expand_dims(self.allproductemb, axis=0),[allproduct_expand_dim_num,1,1])
        self.dis = tf.reduce_mean(tf.square(self.expand_item_emb - self.expand_allproduct_num),axis=1)
        self.ranklist = tf.nn.top_k(-self.dis, params.depth).indices
        #tf.tile(tf.expand_dims(a, axis=1),[1,3])
        self.bool_r = tf.equal(self.ranklist, tf.tile(tf.expand_dims(self.currentid, axis=1),[1,params.depth]))
        self.int_r = tf.cast(self.bool_r, tf.int32)
        self.hit_rate = tf.reduce_sum(tf.reduce_sum(self.int_r, axis=1))
        #self.ranklist = tf.argsort(tf.reduct_mean(tf.square(self.expand_item_emb - self.expand_allproduct_num),axis=1))

    def step(self, session, currentid, itememb, allproductemb):
        input_feed = {}
        input_feed[self.currentid.name] = currentid
        input_feed[self.itememb.name] = itememb
        input_feed[self.allproductemb.name] = allproductemb
        output_feed = [self.hit_rate]
        outputs = session.run(output_feed, input_feed)
        return outputs


def GetItemRankListRandomly(userDictKey, current_product_id):
    allUserList = list(userDictKey)[1:]
    allUserList.pop(current_product_id)
    RankList = random.sample(allUserList,99) + [current_product_id]
    random.shuffle(RankList)
    r = np.equal(RankList[:10], np.array([current_product_id] *10)).astype(int)   # 对位
    per_hr10 = hit_rate(r)
    per_mrr10 = mean_reciprocal_rank(r)
    per_ndcg10 = dcg_at_k(r, 10, 1)

    r = np.equal(RankList[:20], np.array([current_product_id] *20)).astype(int)   # 对位
    per_hr20 = hit_rate(r)
    per_mrr20 = mean_reciprocal_rank(r)
    per_ndcg20 = dcg_at_k(r, 20, 1)

    r = np.equal(RankList[:50], np.array([current_product_id] *50)).astype(int)   # 对位
    per_hr50 = hit_rate(r)
    per_mrr50 = mean_reciprocal_rank(r)
    per_ndcg50 = dcg_at_k(r, 50, 1)
    return per_hr10, per_mrr10, per_ndcg10,     per_hr20, per_mrr20, per_ndcg20,    per_hr50, per_mrr50, per_ndcg50


# AllProductEmb -> [product_size, emb_size] 
def GetItemRankList(AllUserEmb, UserEmb, current_user_id, userDictKey):
    allUserList = list(userDictKey)[1:]
    allUserList.pop(current_user_id)

    RankList = random.sample(allUserList,99) + [current_user_id]
    random.shuffle(RankList)
    current_user_index = RankList.index(current_user_id)

    userCandidates = tf.nn.embedding_lookup(AllUserEmb, RankList ,name="embedding_item")
    sess = tf.compat.v1.Session()
    userCandidatesEmb = sess.run(userCandidates)
 
    expand_dim_num = len(RankList)
    UserEmb = np.tile(UserEmb,(expand_dim_num,1))

    depth = 10
    RankList_index = np.argsort(np.mean(np.square(UserEmb-userCandidatesEmb),axis=1))[0:depth]
    r = np.equal(RankList_index, np.array([current_user_index] *depth)).astype(int)   # 对位
    per_hr10 = hit_rate(r)
    per_mrr10 = mean_reciprocal_rank(r)
    per_ndcg10 = dcg_at_k(r, depth, 1)

    depth = 20
    RankList_index = np.argsort(np.mean(np.square(UserEmb-userCandidatesEmb),axis=1))[0:depth]
    r = np.equal(RankList_index, np.array([current_user_index] *depth)).astype(int)   # 对位
    per_hr20 = hit_rate(r)
    per_mrr20 = mean_reciprocal_rank(r)
    per_ndcg20 = dcg_at_k(r, depth, 1)

    depth = 50
    RankList_index = np.argsort(np.mean(np.square(UserEmb-userCandidatesEmb),axis=1))[0:depth]
    r = np.equal(RankList_index, np.array([current_user_index] *depth)).astype(int)   # 对位
    per_hr50 = hit_rate(r)
    per_mrr50 = mean_reciprocal_rank(r)
    per_ndcg50 = dcg_at_k(r, depth, 1)

    return per_hr10, per_mrr10, per_ndcg10,     per_hr20, per_mrr20, per_ndcg20,    per_hr50, per_mrr50, per_ndcg50

def FindTheRank(AllProductEmb, ProductEmb, current_product_id):
    expand_dim_num = np.shape(AllProductEmb)[0]
    ProductEmb = np.tile(ProductEmb,(expand_dim_num,1))
    RankList = np.argsort(np.mean(np.square(ProductEmb-AllProductEmb),axis=1))
    ValueRankList = np.sort(np.mean(np.square(ProductEmb-AllProductEmb),axis=1))
    r = np.equal(RankList, np.array([current_product_id])).astype(int)
    pos = int(np.argwhere(r == 1))  # r.shape == (130,)
    return pos, ValueRankList[0], ValueRankList[pos], ValueRankList[len(ValueRankList) - 1]

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def mean_reciprocal_rank(r):
    return np.sum(r / np.arange(1, r.size + 1))


def hit_rate(r):
    if (np.sum(r) >= 0.9):
        return 1
    else:
        return 0


