class AuctionData:
    def __init__(self, auctionID):
        self.auctionID = auctionID

        self.UserBidList = []
        self.bidderSet = set()
        self.lastTime = 0.
        self.lastBid = 0.

    def AddBid(self, ReviewInfo):
        if ReviewInfo['bid']> self.lastBid and ReviewInfo['bidtime']> self.lastTime:
            self.lastBid = ReviewInfo['bid']
            self.lastTime = ReviewInfo['bidtime']
            self.bidderSet.update([ReviewInfo['bidderID']])
            BidList = dict()

            BidList['bidderID'] = ReviewInfo['bidderID']
            
            BidList['bidderrate'] = ReviewInfo['bidderrate']

            BidList['bid'] = ReviewInfo['bid']

            BidList['bidtime'] = ReviewInfo['bidtime']

            self.UserBidList.append(BidList)

    def PrintUserInfo(self):
        print('\nAuctionID:%s' % self.auctionID)
        # print('Features:%s' % self.auctionID)
        print('final user info:', [self.finalUserID, self.finalUserrate, self.finalPrice,self.finalTime])
        print('UserBidList:')
        for i in range(1,len(self.UserBidList)):
            print(self.UserBidList[i])
    
    def GetUserBidLen(self):
        return len(self.UserBidList)
    def GetUserLen(self):
        return len(self.bidderSet)

    def SetFinalBid(self,index):
        self.finalUserID = self.UserBidList[index]['bidderID']
        self.finalUserrate = self.UserBidList[index]['bidderrate']
        self.finalPrice = self.UserBidList[index]['bid']
        self.finalTime = self.UserBidList[index]['bidtime']
        # self.UserBidList.pop(index)

    def GetFinalBid(self):   # uid 是元数据集里的id
        return [self.finalUserID, self.finalUserrate, self.finalPrice]
        

    def confirmTarget(self, count):   # uid 是元数据集里的id
        self.SetFinalBid(-1)
        # [self.finalUserID, self.finalUserrate, self.finalPrice, self.finalTime]
        uid, timediff,biddiff = [], [],[]
        for i in range(1,len(self.UserBidList)-1):
            uid.append(self.UserBidList[i]['bidderID'])
            timediff.append(self.UserBidList[i]['bidtime']-self.UserBidList[i-1]['bidtime'])
            biddiff.append(self.UserBidList[i]['bid']-self.UserBidList[i-1]['bid'])
        timeuid = uid[timediff.index(min(timediff))]
        biduid = uid[biddiff.index(max(biddiff))]
        flag1= False
        flag2= False
        if timeuid == self.finalUserID:
            count[0]+=1
            flag1= True
        if biduid == self.finalUserID:
            count[1]+=1
            flag2= True
        if self.finalUserID in uid:
            count[2]+=1
            # self.PrintUserInfo()
#單刷撿漏，最後出現+出價小，100， 1117
        return count