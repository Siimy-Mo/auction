from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed

class Embedding(object):
    def __init__(self, DataSet, params):
        self.userNum = DataSet.userNum[0]       # [7306,1119]
        self.productNum = DataSet.auctionNum[0] # [34342,7452]
        # print(self.productNum)  # 6328
        self.params = params
        self.bidder_listNum = DataSet.max_auction_bids_len
        self.durationNum = DataSet.durationNum  #5
        self.openbidNum = DataSet.openbidNum    #2258
        self.typeNum = DataSet.typeNum          #3


        self.auction2bidderList = DataSet.auction2bidders
        self.maxbidderNum = DataSet.maxbidders

        const = tf.constant_initializer(0.0)
        self.product_linear_w = tf.compat.v1.get_variable('w',
            [params.embed_size, params.embed_size],
            initializer=tf.keras.initializers.glorot_normal()
        )
        self.product_linear_b = tf.compat.v1.get_variable('b', 
            [params.embed_size], 
            initializer=const)

        # Emebdding Layer 编码可训练层
        # User Embedding
        self.useremb = tf.compat.v1.get_variable(
            'userembedding',
            [self.userNum, params.embed_size],
            initializer=tf.keras.initializers.glorot_normal()
        )

        # User External Embedding
        self.user_output_emb = tf.compat.v1.get_variable(
            'useroutputembedding',
            [self.userNum, params.embed_size],
            initializer=tf.keras.initializers.glorot_normal()
        )

        # product embedding
        self.productemb = tf.compat.v1.get_variable(
            'productembedding',
            [self.productNum, params.embed_size],
            initializer=tf.keras.initializers.glorot_normal()
        )
        
        self.productembid = tf.compat.v1.get_variable(
            'productembedding_with1/2',
            [self.productNum, params.embed_size/2],
            initializer=tf.keras.initializers.glorot_normal()
        )

        # ############################# New add #########
        self.durationsemb = tf.compat.v1.get_variable(
            'durationsembedding',
            [self.durationNum, params.embed_size/8],   # Num=5
            initializer=tf.keras.initializers.glorot_normal()
        )

        self.openbidsemb = tf.compat.v1.get_variable(
            'openbidsembedding',
            [self.openbidNum, params.embed_size/4],   # Num=588
            initializer=tf.keras.initializers.glorot_normal()
        )
        
        self.typesemb = tf.compat.v1.get_variable(
            'typesembedding',
            [self.typeNum, params.embed_size/8],   # Num= 3
            initializer=tf.keras.initializers.glorot_normal()
        )

        # Bidder list embedding for auctions
        self.biddersemb = tf.compat.v1.get_variable(
            'biddersembedding',
            [self.bidder_listNum, params.embed_size],   # Num=94
            initializer=tf.keras.initializers.glorot_normal()
        )
        
    # input : uid -> [batch_size, 1]
    # output : useremb -> [batch_size, params.embed_size]
    def GetUserEmbedding(self, uid):
        return tf.nn.embedding_lookup(self.useremb, uid)
    
    # if pid is before_pid_pos,Then: 
        # input : pid -> [batch_size, max_product_len]
        # output: productemb -> [batch_size, max_product_len, params.embed_size]


    def GetProductEmbedding_2d(self, all):
        pid = tf.nn.embedding_lookup(self.productemb, all[:,0])
        type_id = tf.nn.embedding_lookup(self.typesemb, all[:,1])
        openbid_id = tf.nn.embedding_lookup(self.openbidsemb, all[:,2])
        duration_id = tf.nn.embedding_lookup(self.durationsemb, all[:,3])

        # final = tf.concat([pid,type_id], 1)
        return type_id

    def GetProductEmbedding(self, pid):
        return tf.nn.embedding_lookup(self.productemb, pid)

    def testAuctionList(self):
        return self.testAuctionList

    def GetAllProductEmbedding(self):
        return self.productemb

    def GetAllTestProductEmbedding(self):
        return tf.nn.embedding_lookup(self.productemb, self.testAuctionList)

    def GetAllUserEmbedding(self):
        return self.useremb


    def GetProductEmbedding_id(self, id):
        return tf.nn.embedding_lookup(self.productembid, id)
    def GetProductEmbedding_type(self, id):
        return tf.nn.embedding_lookup(self.typesemb, id)
    def GetProductEmbedding_openbid(self, id):
        return tf.nn.embedding_lookup(self.openbidsemb, id)
    def GetProductEmbedding_duration(self, id):
        return tf.nn.embedding_lookup(self.durationsemb, id)