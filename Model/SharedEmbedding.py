from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed

class Embedding(object):
    def __init__(self, DataSet, params):
        self.userNum = DataSet.userNum       # 
        self.productNum = DataSet.auctionNum # 
        self.bidderRates = list(DataSet.uid_2_attrs.values())
        self.itemFeatures = list(DataSet.pid_2_attrs.values())
        # user product 转成 one-hot!!
        self.params = params


        const = tf.constant_initializer(0.0)
        self.useridemb = tf.compat.v1.get_variable('w',
            [self.userNum, params.embed_size-1],
            initializer=tf.keras.initializers.glorot_normal()
        )
        self.product_linear_b = tf.compat.v1.get_variable('b', 
            [params.embed_size], 
            initializer=const)

        # Emebdding Layer 编码可训练层
        # User Embedding
        # UserMatrix=np.identity(self.userNum,dtype=np.bool_)   # 對角矩陣
        ItemMatrix=np.identity(self.productNum,dtype=np.bool_)

        ItemCoMatrix = np.concatenate((ItemMatrix, np.array(self.itemFeatures)),1)
        self.bidderRates = np.array(self.bidderRates).reshape(-1,1) # 升维 -> 2D
        # userCoMatrix = np.concatenate((UserMatrix, self.bidderRates),1)
        self.userrateemb = tf.constant(self.bidderRates,name="userids",dtype=tf.float32)

        self.itememb = tf.constant(ItemCoMatrix,name="itemids",dtype=tf.float32)

        self.allItemCoEmb=tf.compat.v1.layers.dense(inputs=self.itememb, units=params.embed_size,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01))


    # input : uid -> [batch_size, 1]
    # output : useremb -> [batch_size, params.embed_size]
    def GetUserEmbedding(self, uid):
        return tf.nn.embedding_lookup(self.useremb, uid)

    def testAuctionList(self):
        return self.testAuctionList

    def GetAllUserEmbedding(self):
        return self.useremb


    # auction-based function    
    def GetUserEmbedding(self, ids):
        ids = tf.cast(ids, tf.int32)
        idemb = tf.nn.embedding_lookup(self.useridemb, ids ,name="embedding_user")
        rateemb = tf.nn.embedding_lookup(self.userrateemb, ids ,name="embedding_user")
        return tf.concat([idemb,rateemb], axis=-1)

    def GetItemEmbedding(self, ids):
        ids = tf.cast(ids, tf.int32)
        return tf.nn.embedding_lookup(self.allItemCoEmb, ids,name="embedding_item")

    def GetAllUserEmbedding(self):
        return tf.concat([self.useridemb,self.userrateemb], axis=1)
