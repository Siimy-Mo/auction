class BidderData:
    def __init__(self, bidderID, biderrate):
        self.bidderID = bidderID

        self.biderrate = biderrate

        self.UserBidList = []

    def AddBid(self, ReviewInfo):
        BidList = dict()
        BidList['test'] = ReviewInfo['test']
        # ReviewKey = ['test','bidderID', 'asin', 'bidorBuy', 'unixBidTime', 'bidTime']
        BidList['bidderID'] = ReviewInfo['bidderID']

        BidList['asin'] = ReviewInfo['asin']

        BidList['bidorBuy'] = ReviewInfo['bidorBuy']

        BidList['unixBidTime'] = ReviewInfo['unixBidTime']

        BidList['bidTime'] = ReviewInfo['bidTime']
        
        BidList['test'] = ReviewInfo['test']

        self.UserBidList.append(BidList)

    def PrintUserInfo(self):
        print('BidderID:%s' % self.bidderID)
        print('UserBidList:', self.UserBidList)
    
    def GetUserBidLen(self):
        return len(self.UserBidList)