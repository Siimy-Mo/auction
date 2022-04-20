import os,json,argparse
from re import X
from tkinter import Y
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy.interpolate import make_interp_spline

train, eval, test = 95,3,2
# 7,2,1
# 95,3,1
def bar_bid(data):
    bidder2itemDict = dict()
    for index, row in data.iterrows():
        uid = row['bidderID']
        pid = row['asin']
        if uid not in bidder2itemDict:
            bidder2itemDict[uid] = set()
        bidder2itemDict[uid].update([pid])

    bidder2itemLen = []
    maxbids = 0
    for key in bidder2itemDict:
        bids = len(bidder2itemDict[key])
        if bids > maxbids:
            maxbids = bids
        bidder2itemLen.append(bids)

    plt.bar(bidder2itemLen, range(len(bidder2itemLen)), width=0.39)
    plt.xlabel('# bids of each bidder')
    plt.ylabel('# bidders')
    plt.title('Historgram of \nthe number of bids of each bidder')
    plt.legend()
    plt.show()

def bar_buy(data):
    bidder2itemDict = dict()
    for index, row in data.iterrows():
        uid = row['bidderID']
        pid = row['asin']
        flag = row['bidorBuy']
        if uid not in bidder2itemDict:
            bidder2itemDict[uid] = set()
        if flag == 1:
            bidder2itemDict[uid].update([pid])

    bidder2itemLen = []
    maxbids = 0
    for key in bidder2itemDict:
        bids = len(bidder2itemDict[key])
        if bids > maxbids:
            maxbids = bids
        bidder2itemLen.append(bids)

    plt.bar(bidder2itemLen, range(len(bidder2itemLen)), width=0.39)
    plt.xlabel('# buy of each bidder')
    plt.ylabel('# bidders')
    plt.title('Historgram of \nthe number of buy of each bidder')
    plt.legend()
    plt.show()

def bidDistribution(bidRecord):
    print('-----bid percentage by UnixTime-----')
    auctionBid = dict()
    auctionTime = dict()
    auctionType = dict()
    auctionUser = dict()
    TypeColor = set()
    bidRecord = bidRecord[['auctionid','bid','bidtime','bidder','openbid','price','item']]
    bidRecord = bidRecord.sort_values(by=['auctionid','bidtime'], ascending=True)
    print(bidRecord)
    
    # 查询所有竞拍记录，除去无效竞拍【时间递增价格反而递减】
    for index, row in bidRecord.iterrows():
        pid = row['auctionid']
        bid = row['bid']
        openPrice = row['openbid']
        finalPrice = row['price']
        priceUnit = finalPrice - openPrice
        if priceUnit == 0:
            continue

        bidtime = row['bidtime']

        if pid not in auctionBid:
            auctionBid[pid] = []
            auctionTime[pid] = []
            auctionUser[pid] = set()
            auctionType[pid] = row['item']
            lastTime, lastBid = 0,0
            TypeColor.update([row['item']])

        #确保随着时间的增加出价也是增加的，而不是同价或者低价。
        if bidtime>lastTime and (bid-openPrice)/priceUnit>lastBid:
            auctionBid[pid].append((bid-openPrice)/priceUnit)
            auctionTime[pid].append(bidtime)
            auctionUser[pid].update([row['bidder']])

    plt.title('Bid Distribution based on One auction')
    colors=['orange','green','blue']
    TypeColor = list(TypeColor)
    # 要求参竞拍次数>3, 参与人数>2,
    count = 0
    for pid in auctionBid:
        if len(auctionTime[pid])>3 and len(auctionUser[pid]) > 2:
            count += 1
            xTime = auctionTime[pid]
            yBid = np.array(auctionBid[pid])
            # xTime = np.array(xTime/np.max(xTime))
            xTime = np.array(xTime)

            if np.isnan(yBid).all() == False:
                c = colors[TypeColor.index(auctionType[pid])]
                plt.plot(xTime, yBid ,alpha = 0.5, linewidth = 1, color = c )

    plt.xlabel('Time')
    plt.ylabel('Bid percentage')
    plt.show()
    print(count) # 539个
    input()


def main():
    data_path = './Dataset/'
    ReviewDatas = pd.read_csv(data_path+'bidRecord_kaggle.csv', sep=';', engine='python')
    productMeta = pd.read_csv(data_path+'productMeta_kaggle.csv', sep=';', engine='python')
    kaggleAuction = pd.read_csv(data_path+'auction.csv', sep=',', engine='python')
    print('Before:')
    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', productMeta.shape)

    # kaggle dataset processing

    kaggleAuction.bidder=kaggleAuction.bidder.astype(str)

    auctionidEncoder= preprocessing.LabelEncoder()
    BidderEncoder= preprocessing.LabelEncoder()

    kaggleAuction.auctionid= auctionidEncoder.fit_transform(kaggleAuction.auctionid)
    kaggleAuction.bidder= BidderEncoder.fit_transform(kaggleAuction.bidder)

    kaggleAuction.bidder=kaggleAuction.bidder.astype(int)
    kaggleAuction.auctionid=kaggleAuction.auctionid.astype(int)

    # 按时间排序，时间从现在到之前
    itemList = []
    bidder2itemDict = dict()

    # ReviewDatas = ReviewDatas.drop_duplicates().sort_values(by='unixBidTime', ascending=False)
    # bidsList = ReviewDatas[['bidderID','asin','bidorBuy']].drop_duplicates()
    # itemList = set(bidsList[['asin']].drop_duplicates().transpose().values.tolist()[0])

    # bar_bid(bidsList)
    # bar_buy(bidsList)
    bidDistribution(kaggleAuction)

    print('user number: ', len(bidder2itemDict))
    print('item number: ', len(itemList))




if __name__ == "__main__":
    main()