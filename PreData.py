import Data_loader.Data_Generator_auction as data
import Model.SharedEmbedding as SE
import Model.model_test as model_test
import os,json,argparse
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt
import pandas as pd

# train, eval, test = 95,3,2
train, eval, test = 7,2,1
# 7,2,1
# 95,3,1
def main():
    data_path = './Dataset/'
    ReviewDatas = pd.read_csv(data_path+'bidRecord_eBay.csv', sep=';', engine='python')
    productMeta = pd.read_csv(data_path+'productMeta_eBay.csv', sep=';', engine='python')
    print('Before:')
    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', productMeta.shape)

    # 按时间排序，时间从现在到之前
    ReviewDatas = ReviewDatas.sort_values(by='unixBidTime', ascending=False)
    ReviewDatas = ReviewDatas.drop_duplicates(['bidderID','asin','bidorBuy','bid'])   # 10w->5w 104078->49532
    print('\nReviewDatas shape after drop_duplicates: ', ReviewDatas.shape)
    bidsList = ReviewDatas[['bidderID','asin']]
    print('bidsList[bidderID,asin] shape',bidsList.shape)
    testUserLen = int(bidsList.shape[0] / (train+ eval+ test) * test)
    print('testUserLen',testUserLen)

    # 寻找测试集中的有效拍卖会 intersectionList
    bidsList_test = bidsList.head(testUserLen)   # 提取了对应的test期间['bidderID','asin'] 9： 98：946
    bidsList_train = bidsList.tail(bidsList.shape[0]-testUserLen)   # 提取了对应的test期间['bidderID','asin'] 9： 98：946
    if bidsList_test.shape[0] +bidsList_train.shape[0]== bidsList.shape[0]:
        print('bidsList_train / test 切割没问题！')
    trainAuctionList = set(bidsList_train[['asin']].drop_duplicates().transpose().values.tolist()[0])  # 还要筛选进行时和未来时的auction
    testAuctionList = set(bidsList_test[['asin']].drop_duplicates().transpose().values.tolist()[0])  # 还要筛选进行时和未来时的auction
    intersectionList = testAuctionList.intersection(trainAuctionList)
    testOnlyList = testAuctionList.difference(trainAuctionList)
    print('Auction# train/test/intersection',len(trainAuctionList), len(testAuctionList),len(intersectionList),len(testOnlyList)) 
    print('跨线拍卖会数量：',len(intersectionList))

    #  在test pair中找到有效拍卖会的参与者A
    bidderList = set(bidsList_test[['bidderID']].drop_duplicates().transpose().values.tolist()[0])
    print('\ntest bidder # Before',len(bidderList))
    indexList = []
    for index, row in bidsList_test.iterrows():
        bidder = row['bidderID']
        asin = row['asin']
        if asin in intersectionList :
            indexList = indexList + [index]
    # print('bidsList_test中跨线拍卖会的index数量：',len(indexList))
    test_pair = bidsList_test.loc[indexList] 
    valid_bidderList = set(test_pair[['bidderID']].drop_duplicates().transpose().values.tolist()[0])
    print('test bidder # After',len(valid_bidderList)) 

    # 给bid record里添加是否是test期间的tag
    indexDelete = []
    indexList_train, indexList_test = [], []
    testOnlyAuction,i =[],[]
    ReviewDatas = ReviewDatas.reset_index()
    for index, row in ReviewDatas.iterrows():
        bidder = row['bidderID']
        asin = row['asin']
        if index < testUserLen :# 处于test期间的bids
            # valid
            # unvalid -> 删除
            if asin in testOnlyList:    # belongs to FUTURE
                indexDelete = indexDelete + [index]
                testOnlyAuction.append(asin)

            else:
                indexList_test = indexList_test + [index]
                i.append(asin)
        else:   # 处于train期间的bids
            if asin in intersectionList and bidder in valid_bidderList: # 属于要预测的auction且bidder的bids需要删除
                indexDelete = indexDelete + [index]
            else:
                indexList_train = indexList_train + [index]
    print('bids of train/test/ delete',len(indexList_train),len(indexList_test),len(indexDelete))
    ReviewDatas = ReviewDatas.drop(indexDelete)
    ReviewDatas.loc[indexList_test,'test'] = True
    ReviewDatas.loc[indexList_train,'test'] = False
    print(ReviewDatas[ReviewDatas['test']==True].shape,ReviewDatas[ReviewDatas['test']==False].shape , ReviewDatas.shape)
    final_all_auction = set(ReviewDatas[['asin']].drop_duplicates().transpose().values.tolist()[0])

    test_ = ReviewDatas[ReviewDatas['test']==True]
    train_ = ReviewDatas[ReviewDatas['test']==False]
    test_aid = test_[['asin']].drop_duplicates().values.tolist()
    train_aid = train_[['asin']].drop_duplicates().values.tolist()
    print('test bids:', test_.shape[0], 'Auction #',len(test_aid))
    print('train_ bids:', train_.shape[0], 'Auction #',len(train_aid))
    test_bidder_bids, train_bids = 0,0

    for index, row in train_.iterrows():
        bidder = row['bidderID']
        asin = row['asin']
        if [asin] in test_aid:
            test_bidder_bids +=1
        else:
            train_bids +=1
    print(test_bidder_bids,train_bids)


    # 删除A在训练集中对有效拍卖会的拍卖纪录


    print('\nFinal data shape: ', ReviewDatas.shape)
    print('Train/Eval/Test datasets: {0}/{1}/{2}'.format(train, eval, test))
    # print('Test - bidders/auctions/bids #: {0}/{1}/{2}'.format(numbers['target bidder#'], numbers['auction# in test'], numbers['test bid#']))
    # print('For the auctions, train:test is  {0}/{1}. total:{2}'.format(len(auction_all) - numbers['auction# in test'], numbers['auction# in test'],len(auction_all)))
    # print('For the bids, train:test is  {0}/{1}'.format(len(trainbids),numbers['test bid#']))
    # print('For these target bidders {0}, all auctions/bids #: {1}/{2}'.format(numbers['target bidder#'], numbers['all auction#'], numbers['all bid#']))


    indexList = []
    for index, row in productMeta.iterrows():
        asin = row['asin']
        if asin in final_all_auction:
            indexList = indexList + [index]
    MetaDatas= productMeta.loc[indexList,:]


    print('After:')

    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', MetaDatas.shape)
    ReviewDatas.to_csv("./Dataset/Review/bidRecord.csv", sep=';', encoding='utf-8', index=False)
    MetaDatas.to_csv("./Dataset/Meta/productMeta.csv", sep=';', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()