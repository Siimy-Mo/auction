import os,json,argparse
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt
import pandas as pd

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

def main():
    data_path = './Dataset/'
    ReviewDatas = pd.read_csv(data_path+'bidRecord_graph.csv', sep=';', engine='python')
    productMeta = pd.read_csv(data_path+'productMeta_graph.csv', sep=';', engine='python')
    print('Before:')
    print('\nReviewDatas shape: ', ReviewDatas.shape)
    print('MetaDatas shape: ', productMeta.shape)

    # 按时间排序，时间从现在到之前
    itemList = []
    bidder2itemDict = dict()

    ReviewDatas = ReviewDatas.drop_duplicates().sort_values(by='unixBidTime', ascending=False)
    bidsList = ReviewDatas[['bidderID','asin','bidorBuy']].drop_duplicates()
    itemList = set(bidsList[['asin']].drop_duplicates().transpose().values.tolist()[0])

    bar_bid(bidsList)
    bar_buy(bidsList)

    # for uid in bidder2itemLenDict:
    #     bidset =bidder2itemLenDict[uid]

    print('user number: ', len(bidder2itemDict))
    print('item number: ', len(itemList))




if __name__ == "__main__":
    main()