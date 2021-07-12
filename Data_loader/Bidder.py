class BidderData:
    def __init__(self, bidderID):
        self.bidderID = bidderID

        self.UserBidList = []

    def AddBid(self, ReviewInfo):
        BidList = dict()
        # ReviewKey = ['bidderID', 'asin', 'bidorBuy', 'unixBidTime', 'bidTime']
        BidList['bidderID'] = ReviewInfo['bidderID']

        BidList['asin'] = ReviewInfo['asin']

        BidList['bidorBuy'] = ReviewInfo['bidorBuy']

        BidList['unixBidTime'] = ReviewInfo['unixBidTime']

        BidList['bidTime'] = ReviewInfo['bidTime']

        self.UserBidList.append(BidList)

    def PrintUserInfo(self):
        print('BidderID:%s' % self.bidderID)
        print('UserBidList:', self.UserBidList)
    
    def GetUserBidLen(self):
        return len(self.UserBidList)