import Data_loader.Data_Generator_auction as data
import Model.SharedEmbedding as SE
import Model.model_test as model_test
import os,json,argparse
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt
import pandas as pd

train, eval, test = 95,3,2
# 7,2,1
# 95,3,1
def main():
    data_path = './Dataset/'
    ReviewDatas = pd.read_csv(data_path+'bidRecord_graph.csv', sep=';', engine='python')
    productMeta = pd.read_csv(data_path+'productMeta_graph.csv', sep=';', engine='python')
    print('Before:')
    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', productMeta.shape)
    numbers = dict()

    # 按时间排序，时间从现在到之前
    datalen_before = ReviewDatas.shape
    ReviewDatas = ReviewDatas.drop_duplicates().sort_values(by='unixBidTime', ascending=False)
    print("Shape before & after",datalen_before, ReviewDatas.shape) # 不变
    testUserLen = int(ReviewDatas.shape[0] / (train+ eval+ test) * test)
    print('testUserLen',testUserLen)

    testBidsList = ReviewDatas.head(testUserLen)   # 提取了对应的test期间pandas,包含了同用户同拍卖会的多次互动操作
    bidder_item_List = testBidsList[['bidderID','asin']].drop_duplicates()
    print("bidder_item_List.shape",bidder_item_List.shape)

    testAuctionList = set(bidder_item_List[['asin']].drop_duplicates().transpose().values.tolist()[0])
    bidderList = set(bidder_item_List[['bidderID']].drop_duplicates().transpose().values.tolist()[0])
    numbers['auction# in test'] = len(testAuctionList)
    numbers['target bidder#'] = len(bidderList)
    print(bidder_item_List)

    # 给bid record里添加是否是test期间的tag
    indexList, indexList_test = [], []
    for index, row in ReviewDatas.iterrows():
        if index in testBidsList.index.values:
            indexList_test = indexList_test + [index]
        else:
            indexList = indexList + [index]
    print("Test/Train: ", len(indexList_test), len(indexList))
    ReviewDatas.loc[indexList_test,'test'] = True
    ReviewDatas.loc[indexList,'test'] = False

    # 根据bidderList, 基于目标bidder，筛选bid recording，除掉target用户之外的bids + 统计targe用户参加过的auction不除掉其他用户的bids。
    # auction_all = set()
    # indexList_bidder = []
    # trainbids = []
    # testbids = []
    # for index, row in ReviewDatas.iterrows():
    #     bidder = row['bidderID']
    #     asin = row['asin']
    #     if bidder in bidderList:
    #         indexList_bidder = indexList_bidder + [index]
    #         auction_all.update([asin])
    #         if asin in testAuctionList:
    #             testbids.append(index)
    #         else:
    #             trainbids.append(index)
    # ReviewDatas = ReviewDatas.loc[indexList_bidder,:]
    # numbers['test bid#'] = len(testbids)
    # numbers['all bid#'] = ReviewDatas.shape[0]
    # numbers['all auction#'] = ReviewDatas[['asin']].drop_duplicates().shape[0]
    # allAuctionSet = set(ReviewDatas[['asin']].transpose().values.tolist()[0])

    # 统计targe用户参加过的auction，包含其他用户的bids。
    indexList_relatedAuction = []
    indexList_bids = []
    auction_all = set()
    trainbids = []
    testbids = []
    for index, row in ReviewDatas.iterrows():
        bidder = row['bidderID']
        asin = row['asin']
        if bidder in bidderList:
            indexList_relatedAuction = indexList_relatedAuction + [asin]

    for index, row in ReviewDatas.iterrows():
        asin = row['asin']
        if asin in indexList_relatedAuction:
            indexList_bids = indexList_bids + [index]
            auction_all.update([asin])
            if asin in testAuctionList:
                testbids.append(index)
            else:
                trainbids.append(index)
    ReviewDatas = ReviewDatas.loc[indexList_bids,:]
    numbers['test bid#'] = len(testbids)
    numbers['all bid#'] = ReviewDatas.shape[0]
    numbers['all auction#'] = ReviewDatas[['asin']].drop_duplicates().shape[0]

    print('\nFinal data shape: ', ReviewDatas.shape)
    print('Train/Eval/Test datasets: {0}/{1}/{2}'.format(train, eval, test))
    print('Test - bidders/auctions/bids #: {0}/{1}/{2}'.format(numbers['target bidder#'], numbers['auction# in test'], numbers['test bid#']))
    print('For the auctions, train:test is  {0}/{1}'.format(len(auction_all) - numbers['auction# in test'], numbers['auction# in test']))
    print('For the bids, train:test is  {0}/{1}'.format(len(trainbids),numbers['test bid#']))
    print('For these target bidders {0}, all auctions/bids #: {1}/{2}'.format(numbers['target bidder#'], numbers['all auction#'], numbers['all bid#']))


    indexList, indexList_test = [], []
    for index, row in productMeta.iterrows():
        asin = row['asin']
        if asin in auction_all:
            indexList = indexList + [index]
    MetaDatas= productMeta.loc[indexList,:]


    print('After:')

    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', MetaDatas.shape)
    ReviewDatas.to_csv("./Dataset/Review/bidRecord.csv", sep=';', encoding='utf-8', index=False)
    MetaDatas.to_csv("./Dataset/Meta/productMeta.csv", sep=';', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()