class AuctionData:
    def __init__(self, auctionID):
        self.auctionID = auctionID

        self.UserBidList = []

    def AddBid(self, ReviewInfo):
        BidList = dict()

        # ReviewKey = ['test','bidderID', 'asin', 'bidorBuy', 'unixBidTime', 'bidTime']
        BidList['bidderID'] = ReviewInfo['bidderID']
        
        BidList['bidderrate'] = ReviewInfo['bidderrate']

        BidList['bid'] = ReviewInfo['bid']

        # BidList['time'] = ReviewInfo['time']

        self.UserBidList.append(BidList)

    def PrintUserInfo(self):
        print('BidderID:%s' % self.auctionID)
        # print('Features:%s' % self.auctionID)
        print('UserBidList:', self.UserBidList)
    
    def GetUserBidLen(self):
        return len(self.UserBidList)

    def SetFinalBid(self,index):
        self.finalUserID = self.UserBidList[index]['bidderID']
        self.finalUserrate = self.UserBidList[index]['bidderrate']
        self.finalPrice = self.UserBidList[index]['bid']
        self.UserBidList.pop(index)

    def GetFinalBid(self):   # uid 是元数据集里的id
        return [self.finalUserID, self.finalUserrate, self.finalPrice]