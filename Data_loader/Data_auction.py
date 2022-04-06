import gzip,glob,os,random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from collections import Counter

from tensorflow.python.ops.gen_math_ops import Prod
seed = 39
np.random.seed(seed)


class DataSet(object):
    # self, UserData, metaData, neg_sample_num, max_name_len, max_review_len, max_product_len, 
    # savepath, short_term_size, long_term_size = None, weights=True
    def __init__(self, UserData, metaData,neg_sample_num,max_name_len,max_bid_len,max_product_len,savepath,short_term_size,long_term_size):
        self.parameters = dict()
        # UserData: bidderID;asin;reviewText;unixReviewTime;reviewTime
        # metaData: asin;name;openbid;auction_duration;unixEndDate;endDate;bidders;bids
        self.testUserData = dict()
        self.testmetaData = dict()

        # 用户,商品与词统一成v,v与id之间的转换
        self.id_2_v = dict()
        self.v_2_id = dict()

        # 用户与id之间的转换, 重新编码排序
        self.id_2_user = dict()
        self.user_2_id = dict()
        
        # 商品auction与id之间的转换
        self.id_2_product = dict()
        self.product_2_id = dict()

        # 商品auction的duration与id之间的转换
        self.id_2_duration = dict()
        self.duration_2_id = dict()

        # 商品auction的openbid与id之间的转换
        self.id_2_openbid = dict()
        self.openbid_2_id = dict()

        # 商品auction的type与id之间的转换
        self.id_2_type = dict()
        self.type_2_id = dict()

        # 每次auction被哪些人参与了【train期间】，需要计算最大参与人数，用于创建auction emb，这个应该要传给shared emb
        self.auction2bidders = dict()

        # 用户,商品与词统一成v,v与id之间的转换
        self.uid = []
        self.pid = []
        self.userNum = []
        self.auctionNum = []
        self.bidNum = []
        self.product_2_name = dict()

        # 商品名称的词都用id表示
        self.word_2_id = dict()
        self.id_2_word = dict()

        # 时间均值 标准差
        self.timediffAvg = float()
        self.timediffStd = float()

        self.userBids = dict()
        self.userBidsCount = dict()
        self.userPurchases = dict()
        self.bidCoff=set()
        ##########################
        
        # 保存之前的信息
        self.before_pid_pos = dict()
        self.before_pid_bidsCount = dict()
        self.before_pid_flag = dict()
        
        self.nes_weight = []
        self.word_weight = []

        self.neg_sample_num = int(neg_sample_num)
        self.max_name_len = int(max_name_len)
        self.max_product_len = int(max_product_len)
        self.short_term_size = int(short_term_size)
        self.long_term_size = int(long_term_size)
        self.max_auction_bids_len = int()

        self.neg_sample_num = int(neg_sample_num)

        self.init_dict(UserData, metaData)

        self.train_data = []
        self.test_data = []
        self.init_dataset(UserData,metaData,short_term_size,long_term_size)

        # self.timestampList, self.timeintervalList = self.timeDistribution(UserData)
        
        for key in self.parameters:
            print(key,':', self.parameters[key])

    
    # 统计数据数量，将用户和商品的原id分配成新的uid和pid, vid(vector)，遍历数据后决定test timeline
    def init_dict(self, UserData, metaData):    # User 是Class， meta is pandas
        print('\n------ 1 - 未筛选的数据遍历：生成降序的uid及pid, 选出test的auction'+'\n')
        ProductSet = set()
        PurchaseSet = set()
        words = set()
        uid = 0
        vid = 0# 不懂vid拿来干嘛
        pid = 0

        self.product_2_id['<pad>'] = pid    # 这是啥？
        self.id_2_product[pid] = '<pad>'
        vid += 1
        pid += 1
        for index,row in metaData.iterrows():
            p = index
            self.id_2_product[pid] = p
            self.product_2_id[p] = pid
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['nameList']) > 0):
                    self.product_2_name[p] = metaData.loc[p]['nameList']    # 名字库 ['a p l e', 'i p h o n e', 'x r', '6 4 g b']
                    words.update(' '.join(metaData.loc[p]['nameList']).split(' ')) # 增加的是 出现过的单个字符，例如['s', 'm', ')']
            except:
                pass

            # 更新新的product id
            self.id_2_v[vid] = p
            self.v_2_id[p] = vid
            pid += 1
            vid += 1
        
        self.auctionNum.append(pid)


        self.v_2_id['<pad>'] = vid
        self.id_2_v[vid] = '<pad>'
        self.id_2_user['<pad>'] = uid
        self.user_2_id[uid] = '<pad>'
        
        User_bid_len_list = []  # 统计用户平均拍卖记录
        User_bid_time = []      # 統計用戶競拍時間。
        User_purchase_len_list = []
        uid+=1
        test_pid_list = []
        test_uid_list = []
        for i in range(len(UserData)):
            # user id 录入词典
            self.id_2_user[uid] = UserData[i].bidderID
            self.user_2_id[UserData[i].bidderID] = uid
            # bidderID, asin, bidorBuy, unixBidTime, bidTime
            # 每个用户都唯一而且购买长度大于等于10
  
            #更新产品集合
            pids = []
            purchase_pids=[]
            for j in range(len(UserData[i].UserBidList)):
                asin = UserData[i].UserBidList[j]['asin']
                pid = self.product_2_id[asin]

                pids.append(pid)
                User_bid_time.append(UserData[i].UserBidList[j]['unixBidTime'])
                if UserData[i].UserBidList[j]['bidorBuy'] == 1:
                    purchase_pids.append(UserData[i].UserBidList[j]['asin'])

                # 参与特定auction的bidders名单，用于创建auction emb
                if UserData[i].UserBidList[j]['test'] == False:
                    if pid not in self.auction2bidders:
                        self.auction2bidders[pid]=set()
                    self.auction2bidders[pid].update([uid])
                else:
                    test_pid_list.append(pid)
                    test_uid_list.append(uid)



            # 每个用户的购买记录
            ProductSet.update(pids)    # 整理 参与过的拍卖品id
            PurchaseSet.update(purchase_pids)    # 整理 拍中的拍卖品id
            self.userBids[uid] = pids
            self.userBidsCount[uid] = len(UserData[i].UserBidList)
            self.userPurchases[uid] = purchase_pids

            # 统计用户平均拍卖记录
            UserBidLen = len(UserData[i].UserBidList)
            User_bid_len_list.append(UserBidLen)
            UserPurchaseLen = len(purchase_pids)
            User_purchase_len_list.append(UserPurchaseLen)
            uid += 1
        self.maxbidders = 0
        for aid in self.auction2bidders:
            self.auction2bidders[aid] = sorted(self.auction2bidders[aid])
            length = len(self.auction2bidders[aid])
            if length> self.maxbidders:
                self.maxbidders= length
        print('在train期间，一场拍卖会的最高参与人数为：', self.maxbidders)

        self.userNum.append(uid)      # sum the number of user
        self.bidNum.append(np.sum(User_bid_len_list))
        
        self.average_bid_len = np.mean(User_bid_len_list)
        self.max_bid_len = np.max(User_bid_len_list)
        self.max_purchse_len = np.max(User_purchase_len_list)
        print('(未筛选)總用户數:',uid)
        print('出现过的商品数量：',len(ProductSet))

        self.parameters['train number'] = 0  # user, auction
        self.parameters['test number'] = len(set(test_pid_list))
        self.test_pid_list = test_pid_list
        print('test dataset中的拍卖会数量 # is {0}，参与人数: {1}'.format( len(set(test_pid_list)), len(set(test_uid_list))))

        print('用户中最高竞价Bid次数:',np.max(User_bid_len_list), '最低竞价次数:',np.min(User_bid_len_list),\
            '平均竞价次数:',self.average_bid_len)
        print('用户中最高拍中Purchase次数:',np.max(User_purchase_len_list), '最低拍中次数:',np.min(User_purchase_len_list),\
            '平均拍中次数:',np.mean(User_purchase_len_list))
        self.max_unix_time = np.max(User_bid_time)
        self.min_unix_time = np.min(User_bid_time)
        print('最大时间差： ',self.max_unix_time - self.min_unix_time)

        # 计算划分数据集的时间平方差，用于之后的均一化。
        self.timediffAvg = np.average(User_bid_time)
        self.timediffStd = np.std(User_bid_time)
        
        # 统计商品的名字，计算字母权重？？？还有关键词统计
      
        self.nes_weight = np.zeros(self.auctionNum[0])

        wi = 0
        self.word_2_id['<pad>'] = wi
        self.id_2_word[wi] = '<pad>'
        wi += 1
        vid += 1 # from 34342
        for w in words: # length - 137
            if (w==''):
                continue
            self.word_2_id[w] = wi
            self.id_2_word[wi] = w
            self.v_2_id[w] = vid
            self.id_2_v[vid] = w
            wi += 1
            vid += 1
        self.wordNum = wi
        self.word_weight = np.zeros(wi)
        # print('出现过的词类总数(a,b,c,d,1，文字之类的)：',wi)

    def init_dataset(self, UserData, metaData, short_term_size, long_term_size, weights=True):
        try:
            self.data_X = []
            self.init_metaData(UserData, metaData)  # 处理auction的静态特征，还需要修改。
            
            for U in range(len(UserData)):
                uid = self.user_2_id[UserData[U].bidderID]
                # 每个User的数据，方便以后划分数据集
                User_Data_X = []
                UserBidLen = len(UserData[U].UserBidList)
                
                # bid list
                before_pids_pos = []
                before_pids_time= []

                # the query of purchase list


                # 向前补充short_term_size-1个信息
                for k in range(0, short_term_size - 1):
                    before_pids_pos.append([self.product_2_id['<pad>']] + [0,0,0])

                for l in range(1, UserBidLen):
                    # v(i-short_term_size), v(i - windows_size +1),...,v(i-1)  short项个物品
                    # self.testmetaData[pid] = [row['type'],row['name'],row['openbid'],row['auction_duration'],row['endDate']]
                    before_pid_pos = self.product_2_id[UserData[U].UserBidList[l-1]['asin']]
                    before_pid_pos = [before_pid_pos] + self.pid_2_attrs[before_pid_pos]
                    before_pid_time = int(UserData[U].UserBidList[l-1]['unixBidTime'])
                    before_pids_pos.append(before_pid_pos)

                    # query emb -> time emb:
                    try:
                        before_pids_time.append((before_pid_time - self.timediffAvg)/self.timediffStd) # 标准化
                        # before_pids_time.append(np.log(before_pid_time - self.min_unix_time + 1)) # from DeepFM dense features： log(x+1) -> 全是13.0、14.0
                    except:
                        print("User:%s Item:%s has not query!\n" % (str(UserData[U].bidderID), str(UserData[U].UserBidList[l-1]['asin'])))

                    # vi
                    current_pid_pos = self.product_2_id[UserData[U].UserBidList[l]['asin']]
                    current_pid_pos = [current_pid_pos] + self.pid_2_attrs[current_pid_pos]   # [pid, a1,a2,a3] 目前三个attr
                    # current_pid_time = np.log(UserData[U].UserBidList[l]['unixBidTime'] - self.min_unix_time + 1)  # from DeepFM dense features： log(x+1)
                    current_pid_time = (UserData[U].UserBidList[l]['unixBidTime'] - self.timediffAvg)/self.timediffStd  # 标准化
                    current_pid_test = UserData[U].UserBidList[l]['test']
                    if current_pid_pos[0] in self.auction2bidders:
                        current_pid_bidderList = self.auction2bidders[current_pid_pos[0]]
                        current_pid_bidderList = [0] * (self.maxbidders - len(current_pid_bidderList)) + current_pid_bidderList # 补充9
                    else:
                        current_pid_bidderList = [0] * (self.maxbidders)

                    cur_before_pids_pos = before_pids_pos[-short_term_size:]    # 前k個pid
                    cur_before_pids_time = before_pids_time[-short_term_size:]

                    # generate long-term sequence
                    if len(before_pids_pos) < long_term_size:   # 如果拍卖历史小于15，填充对应的0
                        cur_long_before_pids_pos = [[0,0,0,0]] * (long_term_size - len(before_pids_pos)) + before_pids_pos
                        cur_long_before_pids_pos_len = len(before_pids_pos) - (short_term_size - 1) # 有四个填充0，所以要-4

                    else:   # 大于15则，取最后的15位
                        cur_long_before_pids_pos = before_pids_pos[-long_term_size:]
                        cur_long_before_pids_pos_len = len(cur_long_before_pids_pos)    # 15

                    # 计算时间差
                    timediffList = [0,0,0,0]+[current_pid_time - i for i in cur_before_pids_time] # 用于short 的时间差，估计不需要了。
                    cur_before_pids_time_diff = timediffList[-short_term_size:]

                    if l < short_term_size:
                        cur_before_pids_pos_len = l
                    else:
                        cur_before_pids_pos_len = short_term_size
                    self.before_pid_pos[str(uid) + "_" + str(current_pid_pos)] = cur_before_pids_pos


                    # 处理query emb -> time emb:
                    if len(before_pids_time) < self.long_term_size:# 如果time emb数量少于15。这里的+不对啊，源没替换。
                        cur_long_before_time_pos =  [0] * (self.long_term_size - len(before_pids_time)) + before_pids_time # 补充9
                    else:
                        cur_long_before_time_pos = before_pids_time[-self.long_term_size:]

                    User_Data_X.append((uid,\
                                        cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_time_diff,  #之前节点的情报
                                        cur_long_before_pids_pos, cur_long_before_pids_pos_len,cur_long_before_time_pos,  # 添加long term相关情报
                                        current_pid_pos, current_pid_time, current_pid_test,current_pid_bidderList))# 当前节点的情报。
                    # (uid, cur_before_pids_pos, cur_before_pids_pos_len, current_pid_pos, cur_long_before_pids_pos, cur_long_before_pids_pos_len, \
                    #                         Qids_pos, Len_pos, cur_long_before_query_pos, cur_long_before_query_pos_len,\
                    #                         current_text_ids, current_text_Len, product_len_mask, query_len_mask,cur_long_before_query_len_mask))
                        
                self.data_X.append(User_Data_X)
                
            self.auctionList = set()
            # train_auction_list, test_auction_list
            for u in self.data_X:
                u_len = len(u)# u是list
                if u_len>0:
                    for i in range(u_len):
                        auctionid = u[i][-4][0]
                        testFlag = u[i][-2]
                        self.auctionList.update([auctionid])
                        if testFlag:
                            self.test_data.append(u[i])
                        else:
                            self.train_data.append(u[i])
            
            print('train BIDS # is {0}, test BIDS # is {1}'.format(len(self.train_data),len(self.test_data)) )


        except Exception as e:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')
        
        if weights is not False:
            wf = np.power(self.nes_weight, 0.75)
            wf = wf / wf.sum()
            self.weights = wf

    # Qids_pos, Len_pos = self.trans_to_ids(Q_text_array_pos[Qi], self.max_name_len)
    def trans_to_ids(self, query, max_len, weight_cal = True):
        # 比如：['n e w', 'r s i d e n t', 'e v i l', 'v i l a g e']
        query = query.split(' ')
        qids = []
        for w in query:
            if w == '':
                continue
            qids.append(self.word_2_id[w])
            # 统计词频
            if weight_cal:
                self.word_weight[self.word_2_id[w]-1] += 1
        for _ in range(len(qids), max_len):
            qids.append(self.word_2_id['<pad>'])
        return qids, len(query)


    def neg_sample(self, pos_item):
        # current_neg_item, current_neg_word = Dataset.neg_sample(current_pid_pos[i][0])
        # self.auction_attrs[pid] = [pid, type, openbid, duration]
        neg_item = []
        neg_word = []
        neg_sample_list = self.auction_attrs
        neg_sample_len = len(self.auction_attrs) # 33761
        if pos_item[0] in self.auction_attrs.keys():
            pos_item_attr = neg_sample_list.pop(pos_item[0])
            neg_sample_len -=1
        neg_item = random.sample(list(neg_sample_list.values()), self.neg_sample_num)
        return neg_item,neg_word

    def timeDistribution(self, UserData):
        timestampList = []
        intervalList = []
        min = self.parameters['MinMax Unix time'][0]
        for i in range(len(UserData)):
            for j in range(1,len(UserData[i].UserBidList)):
                time = UserData[i].UserBidList[j]['unixBidTime'] - min
                timestampList.append(time)

                if j > 1:
                    interval = time - UserData[i].UserBidList[j-1]['unixBidTime'] + min
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