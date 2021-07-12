class UserData:
    def __init__(self, ReviewerID):
        self.UserID = ReviewerID

        self.UserPurchaseList = []

    def AddPurchase(self, ReviewInfo):
        PurchaseList = dict()
        # ReviewKey = ['reviewerID', 'asin', 'reviewText', 'reviewTime', 'unixReviewTime']
        PurchaseList['asin'] = ReviewInfo['asin']

        PurchaseList['reviewTime'] = ReviewInfo['reviewTime']

        PurchaseList['unixReviewTime'] = ReviewInfo['unixReviewTime']

        PurchaseList['reviewText'] = ReviewInfo['reviewText']

        self.UserPurchaseList.append(PurchaseList)

    def PrintUserInfo(self):
        print('UserID:%s' % self.UserID)
        print('UserPurchaseList:', self.UserPurchaseList)
    
    def GetUserPurchaseLen(self):
        return len(self.UserPurchaseList)












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
    