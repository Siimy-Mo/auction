import gzip,glob,os,random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from collections import Counter

from tensorflow.python.ops.gen_math_ops import Prod
seed = 39
np.random.seed(seed)
RATE = 0.9

class DataSet(object):
    # self, UserData, metaData, neg_sample_num, max_name_len, max_review_len, max_product_len, 
    # savepath, short_term_size, long_term_size = None, weights=True
    def __init__(self, AuctionData, metaData, neg_sample_num, max_auction_bids_len, short_term_size, long_term_size):
        # AuctionData: bidderID;asin;reviewText;unixReviewTime;reviewTime
        # metaData: asin;name;openbid;auction_duration;unixEndDate;endDate;bidders;bids
        
        # 用户,商品与词统一成v,v与id之间的转换
        self.id_2_v = dict()
        self.v_2_id = dict()

        # 用户与id之间的转换, 重新编码排序
        self.id_2_user = dict()
        self.user_2_id = dict()
        self.uid_2_attrs = dict()

        # 商品auction与id之间的转换
        self.id_2_product = dict()
        self.product_2_id = dict()
        self.pid_2_attrs = dict()

        # 用户,商品与词统一成v,v与id之间的转换
        self.uid = []
        self.pid = []
        self.maxBidderrate, self.minBidderrate = 0,500.0
        self.maxBidPrice, self.minBidPrice = 0,500.0


        # 时间均值 标准差
        self.timediffAvg = float()
        self.timediffStd = float()

        ##########################
        
        # 保存之前的信息
        self.before_pid_pos = dict()
        self.before_pid_bidsCount = dict()
        self.before_pid_flag = dict()
        
        self.nes_weight = []
        self.word_weight = []

        self.neg_sample_num = int(neg_sample_num)
        self.short_term_size = int(short_term_size)
        self.long_term_size = int(long_term_size)
        self.max_auction_bids_len = int(max_auction_bids_len)   # 一场拍卖会的竞拍次数maximum

        self.init_dict(AuctionData, metaData)

        self.train_data = []
        self.test_data = []
        self.init_dataset(AuctionData, short_term_size,long_term_size)

    
    # 统计数据数量，将用户和商品的原id分配成新的uid和pid, vid(vector)，遍历数据后决定test timeline
    def init_dict(self, AuctionData, metaData):    # User 是Class， meta is pandas
        print('\n------ 1 - 未筛选的数据遍历：生成降序的uid及pid, 选出test的auction'+'\n')
        uid, pid = 0,0
        self.id_2_user[uid] = '<pad>'
        self.user_2_id['<pad>'] = uid
        self.uid_2_attrs['<pad>'] = [0. for i in range(1)]
        self.id_2_product[pid] = '<pad>'
        self.product_2_id['<pad>'] = pid    # 字典初始化
        self.pid_2_attrs['<pad>'] = [0. for i in range(7)]
        uid += 1
        pid += 1
        

        uid_purchase_dict = dict()
        auctionBidLen_list = []
        auctionBidderLen_list = []
        for i in range(len(AuctionData)):
            # user id 录入词典
            # 拍卖会需要有2个以上才能进行LSTM
            if len(AuctionData[i].UserBidList) >=2:
                self.id_2_product[pid] = AuctionData[i].auctionID
                self.product_2_id[AuctionData[i].auctionID] = pid
    
                # 拍卖会需要有2个以上才能进行LSTM
                finalPrice = -1
                bidderSet = set()
                for j in range(len(AuctionData[i].UserBidList)):
                    bidderID = AuctionData[i].UserBidList[j]['bidderID']
                    bidderrate = AuctionData[i].UserBidList[j]['bidderrate']
                    bid = AuctionData[i].UserBidList[j]['bid']
                    self.setMinMax(bidderrate, bid)
                    if bidderID not in self.user_2_id:  # 录入相关的用户id
                        self.id_2_user[uid] = bidderID
                        self.user_2_id[bidderID] = uid
                        self.uid_2_attrs[uid] = bidderrate
                        uid += 1

                    if bid > finalPrice: # 记录最后一个出价的user
                        finalUid = self.user_2_id[bidderID]
                        finalPrice = bid
                        finalIndex = j
                    bidderSet.update([self.user_2_id[bidderID]])
                # 取出最后一个用户并且加入到属性里,通过GetFinalBid()获得成交用户信息：
                AuctionData[i].SetFinalBid(finalIndex)
                if finalUid not in uid_purchase_dict:
                    uid_purchase_dict[finalUid] = []
                uid_purchase_dict[finalUid].append(pid)

                # 统计1：一场拍卖会的竞拍次数，统计2：一场拍卖会的竞拍者人数
                auctionBidLen_list.append(len(AuctionData[i].UserBidList))
                auctionBidderLen_list.append(len(bidderSet))
                pid += 1
        # 根据pid数量平分pid -> train:test, pid 分出来
        self.train_test_split = pid * RATE
        self.userNum = uid
        self.auctionNum = pid
        self.bidNum = np.sum(auctionBidLen_list)
        print("数据统计:\n")
        print("用户数量：", uid)
        print("拍卖物品数量：", pid)
        print("竞拍记录数量：", self.bidNum)
        bidsMaxNum_perAuction = np.max(auctionBidLen_list)
        bidderMaxNum_perAuction = np.max(auctionBidderLen_list)
        bidsAverageNum_perAuction = np.average(auctionBidLen_list)
        bidderAverageNum_perAuction = np.average(auctionBidderLen_list)
        print('拍卖会中最高竞价次数: %i 以及最高竞价人数为：%i'% (bidsMaxNum_perAuction,bidderMaxNum_perAuction))
        print('拍卖会中平均竞价次数: %i 以及平均竞价人数为：%i'% (bidsAverageNum_perAuction,bidderAverageNum_perAuction))

        self.user_attrs_modification()
        maxBid = self.maxBidPrice - self.minBidPrice

        # 录入相关product 的基础信息：
        for index, row in metaData.iterrows():
            if index in self.product_2_id:
                pid = self.product_2_id[index]
                # 处理attrs
                openbidPrice = (row['openbid']-self.minBidPrice) / maxBid
                # type duration 转 one-hot,都是012
                Onehot_type, Onehot_duration = [0.,0.,0.],[0.,0.,0.]
                Onehot_type[int(float(row['type']))] = 1.0
                Onehot_duration[int(float(row['auction_duration']))] = 1.0
                pid_attrs = [openbidPrice]+Onehot_type+Onehot_duration
                self.pid_2_attrs[pid] = pid_attrs


    def init_dataset(self, AuctionData, short_term_size, long_term_size, weights=True):
        # 将数据处理分类成train+test， 根据长度切割
        try:
            self.data_X = []
            # self.init_metaData(AuctionData, metaData)  # 处理auction的静态特征，还需要修改。


            for U in range(len(AuctionData)):
                if AuctionData[U].auctionID not in self.product_2_id:
                    continue
                pid = self.product_2_id[AuctionData[U].auctionID]

                if pid > self.train_test_split:
                    testSignal = True
                else:
                    testSignal = False

                # 每个User的数据，方便以后划分数据集
                Auction_Data_X = []
                UserBidLen = len(AuctionData[U].UserBidList)
                
                # bid list of bidders
                before_uids_pos = []
                before_uids_bids = []
                # 向前补充short_term_size-1个信息

                for k in range(0, short_term_size - 1):
                    before_uids_pos.append(self.product_2_id['<pad>'])# uid+rate
                    before_uids_bids.append(0.0)# uid+rate

                for l in range(1, UserBidLen):
                    # v(i-short_term_size), v(i - windows_size +1),...,v(i-1)  short项个物品
                    before_uid_pos = self.user_2_id[AuctionData[U].UserBidList[l-1]['bidderID']]
                    before_uid_bid = AuctionData[U].UserBidList[l-1]['bid']
                    before_uids_pos.append(before_uid_pos)
                    before_uids_bids.append(before_uid_bid)# uid+rate

                    # vi
                    current_uid_pos =  self.user_2_id[AuctionData[U].UserBidList[l]['bidderID']]
                    current_uid_bid =  AuctionData[U].UserBidList[l]['bid']

                    # 确认是否是test,test的数据要设置成最后一个用户!

                    cur_before_uids_pos = before_uids_pos[-short_term_size:]    # 前k個pid

                    if l < short_term_size:
                        cur_before_uids_pos_len = l
                    else:
                        cur_before_uids_pos_len = short_term_size
                    
                    if testSignal:
                        if l == UserBidLen-1: # 最后一个user的时候才录入
                            Auction_Data_X.append((testSignal, pid, current_uid_pos,\
                                                cur_before_uids_pos,cur_before_uids_pos_len))
                    else:
                        Auction_Data_X.append((testSignal, pid, current_uid_pos,\
                                            cur_before_uids_pos,cur_before_uids_pos_len))


                self.data_X.append(Auction_Data_X)

            for u in self.data_X:
                u_len = len(u)# u是list
                if u_len>0:
                    for i in range(u_len):
                        testFlag = u[i][0]
                        if testFlag:
                            self.test_data.append(u[i])
                        else:
                            self.train_data.append(u[i])
            
            print('train BIDS # is {0}, test BIDS # is {1}'.format(len(self.train_data),len(self.test_data)) )
            # train BIDS # is 8453, test BIDS # is 58

        except Exception as e:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')
        
        if weights is not False:
            wf = np.power(self.nes_weight, 0.75)
            wf = wf / wf.sum()
            self.weights = wf


    def neg_sample(self, pos_user):
        # current_neg_item, current_neg_word = Dataset.neg_sample(current_pid_pos[i][0])
        # uid_pos = [uid, rate] -> uid_2_attrs[] = rate
        neg_user = []
        neg_sample_list = self.uid_2_attrs.copy()
        neg_sample_list.pop('<pad>')
        neg_sample_len = len(self.uid_2_attrs) # 3371
        if pos_user in self.uid_2_attrs.keys():
            pos_user_attr = neg_sample_list.pop(pos_user)
            neg_sample_len -=1
        neg_user = random.sample(list(neg_sample_list.keys()), self.neg_sample_num)

        return neg_user

    def timeDistribution(self, AuctionData):
        timestampList = []
        intervalList = []
        min = self.parameters['MinMax Unix time'][0]
        for i in range(len(AuctionData)):
            for j in range(1,len(AuctionData[i].UserBidList)):
                time = AuctionData[i].UserBidList[j]['unixBidTime'] - min
                timestampList.append(time)

                if j > 1:
                    interval = time - AuctionData[i].UserBidList[j-1]['unixBidTime'] + min
                    intervalList.append(interval)
        return timestampList, intervalList

    def init_metaData(self,UserData, metaData):
        self.auction_attrs = dict()
        self.auction_list = []
        self.testAuction_attrs = []
        for U in range(len(UserData)):
            uid = self.user_2_id[UserData[U].bidderID]
            self.testUserData[uid] = UserData[U].UserBidList

        # typeRate=dict({'Mobile':[],'Crafts':[],'Game':[]})
        durationNum = 1
        openbidNum = 1
        typeNum = 1
        self.pid_2_attrs = dict()
        self.pid_2_attrs[0]= [0,0,0]  #怎么处理填充的0？
        for index, row in metaData.iterrows():
            pid = self.product_2_id[index]
            self.testmetaData[pid] = [row['type'],row['openbid'],row['auction_duration'],row['endDate'],row['name']]
            type = row['type']
            openbid = row['openbid']
            duration = row['auction_duration']
            
            if duration not in self.duration_2_id:
                self.id_2_duration[durationNum] = duration
                self.duration_2_id[duration] = durationNum
                duration = self.duration_2_id[duration]
                durationNum += 1
            else:
                duration = self.duration_2_id[duration]

            if openbid not in self.openbid_2_id:
                self.id_2_openbid[openbidNum] = openbid
                self.openbid_2_id[openbid] = openbidNum
                openbid = self.openbid_2_id[openbid]
                openbidNum += 1
            else:
                openbid = self.openbid_2_id[openbid]

            if type not in self.type_2_id:
                self.id_2_type[typeNum] = type
                self.type_2_id[type] = typeNum
                type = self.type_2_id[type]
                typeNum += 1
            else:
                type = self.type_2_id[type]

            self.pid_2_attrs[pid] = [type, openbid, duration]
            self.auction_attrs[pid] = [pid, type, openbid, duration]
            if pid in self.test_pid_list:
                self.testAuction_attrs.append([pid, type, openbid, duration])
        self.durationNum = durationNum
        self.openbidNum = openbidNum
        self.typeNum = typeNum

    def predictionReport(self, uid, pred, RankList_10,e):
        pred=pred[0]
        # if uid == 3475:  #竞价5次mobile
        #     print('Target user id: ', uid)
        #     print('Target auction id ', pred)
        #     print('Auction Details: ', self.testmetaData[pred])

        #     print('The bid history of user',uid,'is ')
        #     for i in range(len(self.testUserData[uid])):
        #         pid = self.product_2_id[self.testUserData[uid][i]['asin']]
        #         print(self.testmetaData[pid])

        #     print('Top 10')

        #     for i in range(len(RankList_10)):
        #         print(self.testmetaData[RankList_10[i]])

        if pred in RankList_10:
            # print('Epoch:',e ,'Hit!, Rank: ', RankList_10.index(pred))
            return 1
        return 0  # allSets_auction, auction_testOnly =

        # print('The top-10 auction list is:',RankList_10)
        # Craftnum,Mobile = 0,0
        # for i in range(10):
        #     if self.testmetaData[RankList_10[i]][0] == 'Crafts':
        #         Craftnum+=1
        #     if self.testmetaData[RankList_10[i]][0] == 'Mobile':
        #         Mobile+=1
            # print(self.testmetaData[RankList_10[i]])

        # print('Epoch:',e ,'Number of Craft ', Craftnum, ', Number of Mobile ',Mobile)
    
    def hitList(self, uid, pred, RankList_10,e):
        if pred in RankList_10:
            return [uid]
        else:
            return []

    def setMinMax(self, bidderrate, bid):
        if bidderrate > self.maxBidderrate:
            self.maxBidderrate = bidderrate
        if bid > self.maxBidPrice:
            self.maxBidPrice = bidderrate

        if bidderrate < self.minBidderrate:
            self.minBidderrate = bidderrate
        if bid < self.minBidPrice:
            self.minBidPrice = bidderrate

    def user_attrs_modification(self):
        maxRate =  self.maxBidderrate - self.minBidderrate

        for uid in self.uid_2_attrs.keys():
            newRate = (self.uid_2_attrs[uid]-self.minBidderrate) / maxRate
            self.uid_2_attrs[uid] = newRate
