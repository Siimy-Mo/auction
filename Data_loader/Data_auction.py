import gzip,glob,os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from collections import Counter
seed = 39
np.random.seed(seed)


class DataSet(object):
    # self, UserData, metaData, neg_sample_num, max_name_len, max_review_len, max_product_len, 
    # savepath, short_term_size, long_term_size = None, weights=True
    def __init__(self, UserData, metaData,neg_sample_num,max_name_len,max_bid_len,max_product_len,savepath,short_term_size):

        # UserData: bidderID;asin;reviewText;unixReviewTime;reviewTime
        # metaData: asin;name;openbid;auction_duration;unixEndDate;endDate;bidders;bids

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
        
        # 用户,商品与词统一成v,v与id之间的转换
        self.uid = []
        self.pid = []
        self.product_2_name = dict()

        # 商品名称的词都用id表示
        self.word_2_id = dict()
        self.id_2_word = dict()

        self.userBids = dict()
        self.userBidsCount = dict()
        self.eval_baseline=float()
        self.test_baseline=float()

        self.userPurchases = dict()
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
        self.max_auction_bids_len = int()

        self.neg_sample_num = int(neg_sample_num)

        self.init_dict(UserData, metaData)
        self.auction_dataset = self.init_auction_dict(UserData, metaData)

        self.train_data = []
        self.test_data = []
        self.eval_data = []

        self.init_dataset(UserData ,short_term_size)
        
        # self.dataset_report()
        self.init_sample_table()
        # self.WriteToFile(savepath)
    
   
    def init_dict(self, UserData, metaData):
        ProductSet = set()
        PurchaseSet = set()
        words = set()
        uid = 0
        # 不懂vid拿来干嘛
        vid = 0
        self.v_2_id['<pad>'] = vid
        self.id_2_v[vid] = '<pad>'
        
        
        User_bid_len_list = []  # 统计用户平均拍卖记录
        User_bid_time = []      # 統計用戶競拍時間。
        User_purchase_len_list = []
        for i in range(len(UserData)):
            self.id_2_user[uid] = UserData[i].bidderID
            self.user_2_id[UserData[i].bidderID] = uid

            # bidderID, asin, bidorBuy, unixBidTime, bidTime
            # 每个用户都唯一而且购买长度大于等于10
  
            #更新产品集合
            asins = []
            purchase_asins=[]
            for j in range(len(UserData[i].UserBidList)):
                asins.append(UserData[i].UserBidList[j]['asin'])
                User_bid_time.append(UserData[i].UserBidList[j]['unixBidTime'])
                if UserData[i].UserBidList[j]['bidorBuy'] == 1:
                    purchase_asins.append(UserData[i].UserBidList[j]['asin'])

       
            # 每个用户的购买记录
            ProductSet.update(asins)    # 整理 参与过的拍卖品id
            PurchaseSet.update(purchase_asins)    # 整理 拍中的拍卖品id
            self.userBids[uid] = asins
            self.userBidsCount[uid] = len(UserData[i].UserBidList)
            self.userPurchases[uid] = purchase_asins

            # 统计用户平均拍卖记录
            UserBidLen = len(UserData[i].UserBidList)
            User_bid_len_list.append(UserBidLen)
            UserPurchaseLen = len(purchase_asins)
            User_purchase_len_list.append(UserPurchaseLen)
            uid += 1
        self.userNum = uid      # sum the number of user
        self.bidNum = np.sum(User_bid_len_list)
        
        self.average_bid_len = np.mean(User_bid_len_list)
        self.max_bid_len = np.max(User_bid_len_list)
        self.max_purchse_len = np.max(User_purchase_len_list)
        print('總人數:',uid)
        print('总商品数',len(ProductSet))
        print('最高竞价次数:',np.max(User_bid_len_list), '最低竞价次数:',np.min(User_bid_len_list),\
            '平均竞价次数:',self.average_bid_len)
        print('最高拍中次数:',np.max(User_purchase_len_list), '最低拍中次数:',np.min(User_purchase_len_list),\
            '平均拍中次数:',np.mean(User_purchase_len_list))


        maxtime=np.max(User_bid_time)
        mintime=np.min(User_bid_time)
        self.eval_baseline = (maxtime - mintime) * 0.7 + mintime
        self.test_baseline = (maxtime - mintime) * 0.9 + mintime
        
        
        # 统计商品的名字，计算字母权重？？？还有关键词统计
        pid = 0
        self.product_2_id['<pad>'] = pid    # 这是啥？
        self.id_2_product[pid] = '<pad>'
        vid += 1
        pid += 1
        for p in ProductSet:
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['name']) > 0):
                    self.product_2_name[p] = metaData.loc[p]['name']
                    words.update(' '.join(metaData.loc[p]['name']).split(' '))
                    
            except:
                pass

            # 更新新的product id
            self.id_2_product[pid] = p
            self.product_2_id[p] = pid
            self.id_2_v[vid] = p
            self.v_2_id[p] = vid
            pid += 1
            vid += 1
        
        self.productNum = pid
        self.nes_weight = np.zeros(self.productNum)

        wi = 0
        self.word_2_id['<pad>'] = wi
        self.id_2_word[wi] = '<pad>'
        #self.v_2_id['<wordpad>'] = vid
        #self.id_2_v[vid] = '<wordpad>'
        wi += 1
        vid += 1
        for w in words:
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
        print('出现过的词类总数(a,b,c,d,1，文字之类的)：',wi)
    
    def init_auction_dict(self, UserData, metaData):
        # 數據錄入self.action中
        ProductSet = dict()
        ProductSet[0]= [0,0,0]
        # make the bid record list by [bidderid]
        bidder_bid_list = dict()
        for i in range(len(UserData)):
            bidderID = UserData[i].bidderID
            uid = self.user_2_id[bidderID]
            
            for j in range(len(UserData[i].UserBidList)):
                auctionID = UserData[i].UserBidList[j]['asin']
                if auctionID in bidder_bid_list:
                    bidder_bid_list[auctionID].append(uid)
                else:
                    bidder_bid_list[auctionID] = []
                    bidder_bid_list[auctionID].append(uid)
            
        durationNum = 0
        openbidNum = 0
        auction_bid_len = []
        for i, value in bidder_bid_list.items():
        # collect the basic info of auctionID(not pid):
        
            # item_type = metaData.loc[asin]['query'] # metadata 裡去掉了type，需要重新預處理data
            duration = int(metaData.loc[i]['auction_duration'].split(' ')[0]) # keyerror:21343
            openbid = metaData.loc[i]['openbid']
            pid = self.product_2_id[i]

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

            ProductSet[pid] = [duration, openbid]+bidder_bid_list[i]
            auction_bid_len.append(len(bidder_bid_list[i]))

        # print(np.average(auction_bid_len), np.max(auction_bid_len), np.min(auction_bid_len))
        self.max_auction_bids_len = np.max(auction_bid_len)
        self.durationNum = len(self.duration_2_id)
        self.openbidNum = len(self.openbid_2_id)
        return ProductSet

    def init_dataset(self, UserData, short_term_size, weights=True):
        try:
            self.data_X = []
            print('len(UserData): ',len(UserData))

            # return id only, related auction pid 不是成交auction
            related_user_list,train_auction_list, eval_auction_list, test_auction_list= self.related_info(UserData, self.eval_baseline, self.test_baseline)
            
            for U in range(len(UserData)):
                uid = self.user_2_id[UserData[U].bidderID]
                User_Data_X = []

                UserBidLen = len(UserData[U].UserBidList)

                # bid list
                before_pids_pos = []
                before_pids_flag = []


                # 向前补充short_term_size-1个信息
                for k in range(0, short_term_size - 1):
                    before_pids_pos.append(self.product_2_id['<pad>'])
                    before_pids_flag.append(-1)

                for l in range(1, UserBidLen):
                    # v(i-short_term_size), v(i - windows_size +1),...,v(i-1)  short项个物品
                    before_pid_pos = self.product_2_id[UserData[U].UserBidList[l-1]['asin']]
                    before_bid_flag = int(UserData[U].UserBidList[l-1]['bidorBuy'])
                    before_pids_pos.append(before_pid_pos)
                    before_pids_flag.append(before_bid_flag)

                    # vi
                    current_pid_pos = self.product_2_id[UserData[U].UserBidList[l]['asin']]
                    cur_before_pids_pos = before_pids_pos[-short_term_size:]    # 前k個pid
                    cur_before_pids_flag = before_pids_flag[-short_term_size:]


                    if l < short_term_size:
                        cur_before_pids_pos_len = l
                    else:
                        cur_before_pids_pos_len = short_term_size
                    self.before_pid_pos[str(uid) + "_" + str(current_pid_pos)] = cur_before_pids_pos
                    self.nes_weight[current_pid_pos] += 1

                    # # product_2_query -> name, 商品的名字
                    # try:
                    #     name_array_pos = self.product_2_name[self.id_2_product[current_pid_pos]]
                    # except:
                    #     # vi物品没有名字，不加入数据集【应该不可能
                    #     name_array_pos = []

                    # for Ni in range(len(name_array_pos)):
                    #     try:
                    #         Qids_pos, Len_pos = self.trans_to_ids(name_array_pos[Ni], self.max_name_len)
                    #         # print('Qids_pos, Len_pos',Qids_pos, Len_pos)
                    #     except:
                    #         break
                    #     product_len_mask = [0.] * (short_term_size - cur_before_pids_pos_len) + [1.] * cur_before_pids_pos_len
                    #     name_len_mask = [1.] * Len_pos + [0.] * (self.max_name_len - Len_pos)

                    #     null_query_list = [self.word_2_id['<pad>']] * self.max_name_len
                        
                    # User_Data_X.append((uid, cur_before_pids_pos, cur_before_pids_pos_len, current_pid_pos,\
                    #                     Qids_pos, Len_pos, name_len_mask))
                    User_Data_X.append((uid, cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_flag,\
                                        current_pid_pos))

                self.data_X.append(User_Data_X)
                
            '''
            数据集划分-根据用户进行划分，划分比例为7:2:1
            '''
            for u in self.data_X:
                # u是list
                u_len = len(u)
                if u_len>0:
                    for i in range(u_len):
                        if u[i][0] in related_user_list:
                            asin = self.id_2_product[u[i][4]]
                            if asin in train_auction_list:
                                self.train_data.append(u[i])
                            if asin in eval_auction_list:
                                self.eval_data.append(u[i])
                            if asin in test_auction_list:
                                self.test_data.append(u[i])
                        else:
                            # print(u[i][0],'不属于相关用户，跳过')
                            break
            print('Bid Lengths of three dataset(related auction but not purchase):', \
                len(self.train_data),len(self.eval_data),len(self.test_data))
                    
            # test_user, test_auction =[],[]
            # for i in range(len(self.eval_data)):
            #     test_user.append(self.eval_data[i][0])
            #     test_auction.append(self.eval_data[i][3])
            # print('Lengths: ',len(set(test_user)),len(set(test_auction)))

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

    def related_info(self, UserData, eval_baseline, test_baseline ):
        train_user_set = []             # uid ; [the count of auctions] - just count, without asin
        eval_user_set= []
        test_user_set= []
        train_auction_list = set()         # auction aisn
        eval_auction_list = set()
        test_auction_list = set()

        allSets_user = []        # The user both in train set and test set

        for i in range(len(UserData)):
            uid = self.user_2_id[UserData[i].bidderID]

            # User_bid_train_list = []
            # User_bid_eval_list = []
            # User_bid_test_list = []
            # 根據時間，分類 數據集
            for j in range(len(UserData[i].UserBidList)):
                if (UserData[i].UserBidList[j]['unixBidTime']) > eval_baseline:
                    if (UserData[i].UserBidList[j]['unixBidTime']) > test_baseline :
                        test_user_set.append(uid)
                        # User_bid_test_list.append(UserData[i].UserBidList[j]['asin'])
                        # if UserData[i].UserBidList[j]['bidorBuy'] == 1:
                        test_auction_list.update([UserData[i].UserBidList[j]['asin']])
                    else:
                        eval_user_set.append(uid)
                        # User_bid_eval_list.append(UserData[i].UserBidList[j]['asin'])
                        # if UserData[i].UserBidList[j]['bidorBuy'] == 1:
                        eval_auction_list.update([UserData[i].UserBidList[j]['asin']])
                else:
                    train_user_set.append(uid)
                    # User_bid_train_list.append(UserData[i].UserBidList[j]['asin'])
                    # if UserData[i].UserBidList[j]['bidorBuy'] == 1:
                    train_auction_list.update([UserData[i].UserBidList[j]['asin']])


        # print('test dataset 的auction 數量：',len(test_auction_list))
        # print('train dataset 的auction 數量：',len(train_auction_list))
        # print('eval dataset 的auction 數量：',len(eval_auction_list))

        print('test dataset 的user 數量：',len(set(test_user_set)))
        print('train dataset 的user 數量：',len(set(train_user_set)))
        print('eval dataset 的user 數量：',len(set(eval_user_set)))

        '''
        数据集划分-根据用户进行划分，划分比例为7:2:1
        '''
        for user in test_user_set:
            if user in train_user_set or user in eval_user_set:
                allSets_user.append(user)
        print('同時存在於test & train 的用戶數量：',len(set(allSets_user)))
        
        related_train_auction_list = []
        related_eval_auction_list = []
        related_test_auction_list = []
        for i in range(len(UserData)):
            uid = self.user_2_id[UserData[i].bidderID]
            if uid in allSets_user:
                if uid in allSets_user:
                    for j in range(len(UserData[i].UserBidList)):
                        asign = UserData[i].UserBidList[j]['asin']
                        if asign in train_auction_list:
                            related_train_auction_list.append(asign)
                        if asign in eval_auction_list:
                            related_eval_auction_list.append(asign)
                        if asign in test_auction_list:
                            related_test_auction_list.append(asign)
                    

        print('train dataset 的auction num：',len(set(related_train_auction_list)))
        print('eval dataset 的auction num：',len(set(related_eval_auction_list)))
        print('test dataset 的auction num：',len(set(related_test_auction_list)))

        # related_auctionNum = related_test_auction_list
        # print('related auction number：',len(set(related_auctionNum)))

        return list(set(allSets_user)), list(set(train_auction_list)), list(set(eval_auction_list)), list(set(test_auction_list))



    def dataset_report(self):
        print('self.train_data', len(self.train_data))
        print('self.eval_data',len(self.eval_data))
        print('self.test_data',len(self.test_data))
        finaldata= self.train_data+self.eval_data+self.test_data
        refer_user=[]
        refer_auction=[]
        for i in range(len(finaldata)):
            refer_user.append(finaldata[i]['bidderID'])
            refer_auction.append(finaldata[i]['asin'])
        
        self.userNum = len(set(refer_user))
        self.purchaseNum = len(set(refer_auction))
        self.bidNum = len(finaldata)
        print('Final length of related user: ', self.userNum)
        print('Final length of related auction: ', self.purchaseNum)
        x = []
        for i in range(len(self.test_data)):
            if self.test_data[i]['asin'] in refer_auction:
                if self.test_data[i]['bidorBuy'] == 1:
                    x.append(self.test_data[i]['asin'])
        print('the # of related auction in test: ', len(set(x)))

    def calculate_test_time(UserData):
        print(UserData)


    def neg_sample(self):
        neg_item = []
        neg_word = []
        for ii in range(self.neg_sample_num):
            neg_item.append(self.sample_table_item[np.random.randint(self.table_len_item)])
            # neg_word.append(self.sample_table_word[np.random.randint(self.table_len_word)])
        #return self.Tran_Pid_2_vid(neg_item), self.Tran_Wid_2_vid(neg_word)
        return neg_item,neg_word


    def init_sample_table(self):
        table_size = 1e6
        count = np.round(self.weights*table_size)
        print(self.weights)
        print(count)
        self.sample_table_item = []
        for idx, x in enumerate(count):
            self.sample_table_item += [idx]*int(x)
        self.table_len_item = len(self.sample_table_item)

        table_size = 1e-2
        count = np.round(self.word_weight*table_size)
        # print(self.word_weight)
        # print(count)
        self.sample_table_word = []
        for idx, x in enumerate(count):
            self.sample_table_word += [idx]*int(x)
        self.table_len_word = len(self.sample_table_word)
