import os,json,argparse

import matplotlib.pyplot as plt
import pandas as pd

# train, eval, test = 95,3,2
train, eval, test = 7,2,1
# 7,2,1
# 95,3,1
def main():
    data_path = './Dataset/'
    ReviewDatas_k = pd.read_csv(data_path+'bidRecord_kaggle.csv', sep=';', engine='python')
    productMeta_k = pd.read_csv(data_path+'productMeta_kaggle.csv', sep=';', engine='python')
    ReviewDatas_e = pd.read_csv(data_path+'bidRecord_eBay.csv', sep=';', engine='python')
    productMeta_e = pd.read_csv(data_path+'productMeta_eBay.csv', sep=';', engine='python')
    print('kaggle\'s dataset:')
    print('\nReviewDatas shape: ', ReviewDatas_k.shape)
    print('MetaDatas shape: ', productMeta_k.shape)
    print('eBay\'s dataset:')
    print('\nReviewDatas shape: ', ReviewDatas_e.shape)
    print('MetaDatas shape: ', productMeta_e.shape)

    # 有两个user 拍中一个item的记录 -> 删除 data
    df_kaggle=ReviewDatas_k.merge(productMeta_k,left_on="asin",right_on="asin",how="outer")  #極大dataframe
    df_eBay=ReviewDatas_e.merge(productMeta_e,left_on="asin",right_on="asin",how="outer")  #極大dataframe

    df_buy = df_kaggle[df_kaggle['bid'] == df_kaggle['price']]# (10624, 8) (656, 2)
    df_item2user = df_buy[['bidderID','asin']].groupby('asin').count().sort_values(by='bidderID')
    df_item2user = df_item2user[df_item2user['bidderID'] >1].index.tolist() # 无效item的pid, name = asin
    drop_index = []
    for index,row in df_kaggle.iterrows():
        asin = row['asin']
        if asin in df_item2user:
            drop_index.append(index)
    df_kaggle = df_kaggle.drop(drop_index).reset_index()

    # df_buy = df_eBay[df_eBay['bid'] == df_eBay['price_x']]
    # df_item2user = df_buy[['bidderID','asin']].groupby('asin').count().sort_values(by='bidderID')
    # df_item2user = df_item2user[df_item2user['bidderID'] >1].index.tolist()
    # print(df_item2user)
    # drop_index = []
    # for index,row in df_eBay.iterrows():
    #     asin = row['asin']
    #     if asin in df_item2user:
    #         drop_index.append(index)
    # df_eBay = df_eBay.drop(drop_index).reset_index()

    # item user bid #:
    print('\nItem, User, Bid # : ')
    print('Kaggle: ', ReviewDatas_k[['asin']].drop_duplicates().shape,ReviewDatas_k[['bidderID']].drop_duplicates().shape,ReviewDatas_k.drop_duplicates().shape)
    print('eBay: ', ReviewDatas_e[['asin']].drop_duplicates().shape,ReviewDatas_e[['bidderID']].drop_duplicates().shape,ReviewDatas_e.drop_duplicates().shape)
    print('\nItem, User, Bid # : (除去1 item for multi users 的无效物品后)')
    print('Kaggle: ', df_kaggle[['asin']].drop_duplicates().shape,df_kaggle[['bidderID']].drop_duplicates().shape,df_kaggle.drop_duplicates().shape)
    print('eBay: ', df_eBay[['asin']].drop_duplicates().shape,df_eBay[['bidderID']].drop_duplicates().shape,df_eBay.drop_duplicates().shape)

    print('\nItem type:')
    print('Kaggle:\n ', df_kaggle[['type']].drop_duplicates())
    print('eBay: \n', df_eBay[['type']].drop_duplicates())


    # each # based on the type:
    print('-------- kaggle --------')
    df_typeName = df_kaggle[['type']].drop_duplicates()
    bidNum = dict()
    itemNum = dict()
    for index,row in df_typeName.iterrows():
        typeName = row['type']
        bidNum[typeName] = df_kaggle[df_kaggle['type'] == typeName].shape[0]
        itemNum[typeName] = df_kaggle[df_kaggle['type'] == typeName].drop_duplicates('asin').shape[0]
    print('item number of each Type',itemNum)
    print('bid number of each Type',bidNum)

    # each # based on the type:
    print('-------- eBay --------')
    df_typeName = df_eBay[['type']].drop_duplicates()
    bidNum = dict()
    itemNum = dict()
    for index,row in df_typeName.iterrows():
        typeName = row['type']
        bidNum[typeName] = df_eBay[df_eBay['type'] == typeName].shape[0]
        itemNum[typeName] = df_eBay[df_eBay['type'] == typeName].drop_duplicates('asin').shape[0]
    print('item number of each Type',itemNum)
    print('bid number of each Type',bidNum)

    # 购买(buy)物品数量 直方图
    df_buy = df_kaggle[df_kaggle['bid'] == df_kaggle['price']] # shape = 596, with condition: bid = price
    df = df_buy[['bidderID','asin']].groupby('bidderID').count().sort_values(by='asin')   # shape = 571, output中的asin意味着此user参加过的items数量
    df_bidder = df[df['asin']>1]
    bidderList = df_bidder.index.tolist()

    keep_index = []
    for index,row in df_buy.iterrows():
        uid = row['bidderID']
        if uid in bidderList:
            keep_index.append(index)
    df_buy2bidder = df_buy.loc[keep_index,:].reset_index()

    df = df_buy2bidder[['bidderID','type']].drop_duplicates().groupby('type').count()
    print(df)
    df = df_buy2bidder[['bidderID','bidderrate','type']].drop_duplicates()
    print(df)
    df['bidderQuality'] = df.index.to_frame()
    df = df.groupby('asin').count()
    x, y = df.index.tolist(), df.values.tolist()
    y = [row[0] for row in y]
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    df_buy = df_eBay[df_eBay['bid'] == df_eBay['price_x']] # shape = 596, with condition: bid = price
    df = df_buy[['bidderID','asin']].groupby('bidderID').count().sort_values(by='asin')   # shape = 571, output中的asin意味着此user参加过的items数量
    df['bidderQuality'] = df.index.to_frame()
    df = df.groupby('asin').count()
    x, y = df.index.tolist(), df.values.tolist()
    y = [row[0] for row in y]
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    # 参与(bid)物品数量 直方图
    df_buy = df_kaggle[df_kaggle['bid'] != df_kaggle['price']] # shape = 596, with condition: bid = price
    df = df_buy[['bidderID','asin']].groupby('bidderID').count().sort_values(by='asin')   # shape = 571, output中的asin意味着此user参加过的items数量
    df['bidderQuality'] = df.index.to_frame()
    df = df.groupby('asin').count()
    x, y = df.index.tolist(), df.values.tolist()
    y = [row[0] for row in y]
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    df_buy = df_eBay[df_eBay['bid'] != df_eBay['price_x']] # shape = 596, with condition: bid = price
    df = df_buy[['bidderID','asin']].groupby('bidderID').count().sort_values(by='asin')   # shape = 571, output中的asin意味着此user参加过的items数量
    df['bidderQuality'] = df.index.to_frame()
    df = df.groupby('asin').count()
    x, y = df.index.tolist(), df.values.tolist()
    y = [row[0] for row in y]
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    # for eBay = 
    # bid history的多样性: Diversity of bided item types
    df_buy = df_eBay[df_eBay['bid'] == df_eBay['price_x']]
    df_buy = df_buy[['bidderID','type']].drop_duplicates()
    df = df_buy.groupby('bidderID').count().sort_values(by='type')
    df['bidderQuality'] = df.index.to_frame()
    df = df.groupby('type').count()
    x, y = df.index.tolist(), df.values.tolist()
    y = [row[0] for row in y]
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item type quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    df_buy = df_eBay[df_eBay['bid'] == df_eBay['price_x']].drop_duplicates().sort_values(by='unixBidTime').reset_index()
    bidderBuyType = dict()
    for index,row in df_buy.iterrows():
        uid = row['bidderID']
        type = row['type']
        if uid not in bidderBuyType:
            bidderBuyType[uid] = []
        bidderBuyType[uid].append(type)
    
    greaterthan1 = 0
    sameToLast= 0
    for uid in bidderBuyType:
        itemList = bidderBuyType[uid]
        if len(itemList) > 1:
            greaterthan1+=1
            if itemList[-1] == itemList[-2]:
                sameToLast+=1

    print('The buy list greater than 1 item', greaterthan1)
    print('The final item type is same to the Last one', sameToLast)
    # print(x,y)
    # plt.bar(x,y, width=0.4,color='blue')
    # plt.xlabel('the item type quality of one user buy')
    # plt.ylabel('amount of user')
    # plt.show()
    # input()

    # df_buy = df_kaggle[df_kaggle['bid'] == df_kaggle['price']] # shape = 596, with condition: bid = price
    # df = df_buy[['bidderID','asin']].groupby('bidderID').count().sort_values(by='asin')   # shape = 571, output中的asin意味着此user参加过的items数量
    # df['bidderId'] = df.index.to_frame()# bidderid = 2831, 有4个购买item
    # df_target = df_kaggle[df_kaggle['bidderID'] == 2831]
    # print(df_target)





if __name__ == "__main__":
    main()