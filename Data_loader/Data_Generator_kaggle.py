import Data_loader.Data_auction_1 as Data
from Data_loader.Auction import AuctionData
from Data_loader.Bidder import BidderData
from sklearn import preprocessing
import os,pickle,errno
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# pd.set_option('display.max_rows',None)

def ReadRawDataFile_e(Filepath):
	bidRecords = pd.read_csv(Filepath, sep=';', engine='python')
	bidRecords.dropna(how='any')
	print(bidRecords.drop_duplicates().shape)
	print(bidRecords['bidderID'].drop_duplicates().shape)
	print(bidRecords['asin'].drop_duplicates().shape)
	bidRecords.columns = ['bidderID','bidderrate','asin','bid','price','bidorBuy','bidtime','DatebidTime']
	bidRecords.bidderID=bidRecords.bidderID.astype(str)

	auctionidEncoder= preprocessing.LabelEncoder()
	BidderEncoder= preprocessing.LabelEncoder()
	itemTypeEncoder= preprocessing.LabelEncoder()
	durationEncoder= preprocessing.LabelEncoder()

	bidRecords.asin= auctionidEncoder.fit_transform(bidRecords.asin)
	bidRecords.bidderID= BidderEncoder.fit_transform(bidRecords.bidderID)
	# bidRecords.item= itemTypeEncoder.fit_transform(bidRecords.item)
	# bidRecords.auction_type= durationEncoder.fit_transform(bidRecords.auction_type)

	bidRecords.bidderID=bidRecords.bidderID.astype(int)
	bidRecords.asin=bidRecords.asin.astype(int)

	# bidRecords.columns = ['asin','bid','bidtime','bidderID','bidderrate','openbid','price','item','auction_type']
	bidRecords = bidRecords.sort_values(by=['asin','bid'], ascending=True) # 竞拍价格从小到大
	bidRecords.dropna(axis=0,how='any')
    # Save Data
	print('BidderRecordDatas:\n',bidRecords)

	return bidRecords

def ReadRawDataFile(Filepath):
	kaggleAuction = pd.read_csv(Filepath, sep=',', engine='python')
	kaggleAuction.dropna(how='any')
	# print(kaggleAuction)
	# print(kaggleAuction.drop_duplicates().shape)
	# print(kaggleAuction['bidder'].drop_duplicates().shape)
	# print(kaggleAuction['auctionid'].drop_duplicates().shape)
	kaggleAuction.bidder=kaggleAuction.bidder.astype(str)

	auctionidEncoder= preprocessing.LabelEncoder()
	BidderEncoder= preprocessing.LabelEncoder()
	itemTypeEncoder= preprocessing.LabelEncoder()
	durationEncoder= preprocessing.LabelEncoder()

	kaggleAuction.auctionid= auctionidEncoder.fit_transform(kaggleAuction.auctionid)
	kaggleAuction.bidder= BidderEncoder.fit_transform(kaggleAuction.bidder)
	kaggleAuction.item= itemTypeEncoder.fit_transform(kaggleAuction.item)
	kaggleAuction.auction_type= durationEncoder.fit_transform(kaggleAuction.auction_type)

	kaggleAuction.bidder=kaggleAuction.bidder.astype(int)
	kaggleAuction.auctionid=kaggleAuction.auctionid.astype(int)
	
	kaggleAuction.columns = ['asin','bid','bidtime','bidderID','bidderrate','openbid','price','item','auction_type']
	kaggleAuction = kaggleAuction.sort_values(by=['asin','bidtime'], ascending=True) # 竞拍价格从小到大
	kaggleAuction.dropna(axis=0,how='any')
    # Save Data
	print('BidderRecordDatas:\n',kaggleAuction)

	return kaggleAuction


def itemAggregation(BidData):
	itemList = []
	item_index = []
	for index, row in BidData.iterrows():
		pid = row['asin']
		if pid not in item_index:
			Auction = AuctionData(row['asin'])
			Auction.AddBid(row)
			itemList.append(Auction)
			item_index.append(pid)
		else:
			index = item_index.index(pid)
			itemList[index].AddBid(row)
	return itemList

def GetMaxBidLen(data):
    maxnum = 0
    for id in data:
        bidNum = id.GetUserBidLen()
        if bidNum > maxnum:
            maxnum = bidNum
    return maxnum


def ReadMetaPart_kaggle(Filepath):
	df = pd.read_csv(Filepath, sep=';', engine='python')
	df.dropna(how='any')
	# asin;type;name;openbid;price;auction_duration;unixEndDate;endDate;bidders;bids
	values = []
	df = df[['asin','openbid','type','auction_duration']].drop_duplicates()
	df.drop_duplicates()
	for index, row in df.iterrows():
		values.append({'asin':row['asin'],'openbid':row['openbid'],'type':row['type'],'auction_duration':row['auction_duration']})
	
	return values

def ReadMetaPart_ebay(ReviewDatas):
	values = []
	ReviewDatas = ReviewDatas[['asin','openbid','item','auction_type']].drop_duplicates()
	print(ReviewDatas)
	ReviewDatas.drop_duplicates()
	for index, row in ReviewDatas.iterrows():
		values.append({'asin':row['asin'],'openbid':row['openbid'],'type':row['item'],'auction_duration':row['auction_type']})
	
	return values
def checkRecord(auctions):
	greaterthan2, count = 0,[0,0,0]
	bidTimes = 0
	length = []
	for i in range(len(auctions)):
		if auctions[i].GetUserBidLen() > 3 and auctions[i].GetUserLen() > 1:
			greaterthan2+=1
			length.append(auctions[i].GetUserBidLen())
			auctions[i].SetFinalBid(-1)
			count = auctions[i].confirmTarget(count)
		# else:
		# 	auctions[i].PrintUserInfo()
	print(len(auctions), greaterthan2, sum(length)/len(length))
	print(count, count[0]+count[1])


# def PreProcessData(params):
# 	print("Start PreProcess Dataset!\n")
# 	# Read the review dataset
# 	BidFilename_MaxBidLen_Dict = BidRecordDataProcess.DoneAllFile(BidDataFilePath=params.ReviewDataPath, BidDataFileProcessInfoPath=params.ProcessInfoPath, BidDataBinSavepath=params.ReviewDataSavepath, BidUserBinSavePath=params.Review_UserDataSavepath)

# 	# # Combine the User from all the review dataset, 這裡用了bid，BidderData中還沒有追加Buy的product
# 	BidAuctionCombineData,max_auction_bids_len = CombineBidData.CombineBidDataSetByUser(BidUserBinSavePath=params.Review_UserDataSavepath, CombineBidUserBinProcessPath=params.ProcessInfoPath,CombineBidUserBinPath=params.Review_CombineUserDataSavepath)

# 	# # Read the meta dataset
# 	MetaFilename_MaxItemLen_Dict = MetaDataProcess.DoneAllFile(MetaDataFilePath=params.MetaDataPath, MetaDataFileProcessInfopath=params.ProcessInfoPath,MetaDataBinSavePath=params.MetaDataSavepath)

# 	# # Combine all the meta dataset
# 	MetaCombineData = CombineMetaData.CombineMetaDataSet(MetaDataBinSavePath=params.MetaDataSavepath, CombineMetaDataBinProcessInfoPath=params.ProcessInfoPath,CombineMetaDataBinSavePath=params.Meta_CombineDataSavepath)

# 	# max_bid_len = max(zip(BidFilename_MaxBidLen_Dict.values(), BidFilename_MaxBidLen_Dict.keys()))[0] 
# 	# max_name_len = max(zip(MetaFilename_MaxItemLen_Dict.values(), MetaFilename_MaxItemLen_Dict.keys()))[0] 

# 	# Create dataset
# 	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
# 	print('DataSetSavePath',DataSetSavePath)

# 	# 1是dict，2是df
# 	print(params.neg_sample_num, max_auction_bids_len, DataSetSavePath, params.short_term_size, params.long_term_size)
# 	Dataset = Data.DataSet(BidAuctionCombineData,MetaCombineData, params.neg_sample_num, max_auction_bids_len, params.short_term_size, params.long_term_size)


# 	print("End PreProcess Dataset!\n")
# 	return Dataset

def Generate_Data(params):
	# kaggleAuction = ReadRawDataFile('./Dataset/auction.csv')
	# metaProduct = pd.DataFrame(ReadMetaPart_kaggle(kaggleAuction))
	# AuctionList = itemAggregation(kaggleAuction)

	eBayItem = ReadRawDataFile_e('./Dataset/bidRecord_eBay2.csv')
	metaProduct = pd.DataFrame(ReadMetaPart_kaggle('./Dataset/productMeta_eBay2.csv'))
	AuctionList = itemAggregation(eBayItem)
	# checkRecord(AuctionList)
	max_auction_bids_len = int(GetMaxBidLen(AuctionList))

	Dataset = Data.DataSet(AuctionList,metaProduct,  params.neg_sample_num, params.predict_neg_num, max_auction_bids_len, params.short_term_size, params.long_term_size)
	return Dataset
	
	# train_startpos += params.batch_size 應該是目前所在位置，+ size， 形成一個範圍。
def Get_next_batch(Dataset, dataset, startpos, batch_size):
	dataset_len = len(dataset)
	CNIList = []
	test_user_probs = []
	if (startpos + batch_size) > dataset_len:
		# Auction_Data_X.append((testSignal, pid,pid_attrs, current_uid_pos,\
        #                                         cur_before_uids_pos,cur_before_uids_pos_len))
		pids = [dataset[i,1] for i in range(startpos, dataset_len)]
		current_uid_pos = [dataset[i,2] for i in range(startpos, dataset_len)]
		current_uid_history = [dataset[i,3] for i in range(startpos, dataset_len)]
		current_bidTime = [dataset[i,4] for i in range(startpos, dataset_len)]
		short_before_uid_pos = [dataset[i,5] for i in range(startpos, dataset_len)]
		short_before_uid_history = [dataset[i,6] for i in range(startpos, dataset_len)]
		cur_before_bidTime = [dataset[i,7] for i in range(startpos, dataset_len)]
		short_before_uid_pos_len = [dataset[i,8] for i in range(startpos, dataset_len)]
		for i in range(dataset_len - startpos):
			user_prob = [0 for i in range(100)]
			current_neg_user = Dataset.neg_sample(current_uid_pos[i])
			allusers = [current_uid_pos[i]] + current_neg_user
			CNIList.append(current_neg_user)

			random.shuffle(allusers)
			user_prob[allusers.index(current_uid_pos[i])] = 1
			test_user_probs.append(user_prob)
			if current_uid_pos[i] in CNIList[i]:
				print(current_uid_pos[i],CNIList[i])
	else:
		pids = [dataset[i,1] for i in range(startpos, startpos + batch_size)]
		current_uid_pos = [dataset[i,2] for i in range(startpos, startpos + batch_size)]
		current_uid_history = [dataset[i,3] for i in range(startpos, startpos + batch_size)]
		current_bidTime = [dataset[i,4] for i in range(startpos, startpos + batch_size)]
		short_before_uid_pos = [dataset[i,5] for i in range(startpos, startpos + batch_size)]
		short_before_uid_history = [dataset[i,6] for i in range(startpos, startpos + batch_size)]
		cur_before_bidTime = [dataset[i,7] for i in range(startpos, startpos + batch_size)]
		short_before_uid_pos_len = [dataset[i,8] for i in range(startpos, startpos + batch_size)]
		for i in range(batch_size):
			user_prob = [0 for i in range(100)]
			current_neg_user = Dataset.neg_sample(current_uid_pos[i])
			allusers = [current_uid_pos[i]] + current_neg_user
			CNIList.append(current_neg_user)

			random.shuffle(allusers)
			user_prob[allusers.index(current_uid_pos[i])] = 1
			test_user_probs.append(user_prob)
			if current_uid_pos[i] in CNIList[i]:
				print(current_uid_pos[i],CNIList[i])

	return np.array(pids), np.array(current_uid_pos),np.array(current_uid_history),np.array(current_bidTime), \
			np.array(short_before_uid_pos),np.array(short_before_uid_history),np.array(cur_before_bidTime),np.array(short_before_uid_pos_len),np.array(CNIList),np.array(test_uid),np.array(test_user_probs)

# Test
#if __name__ == "__main__":
#	PreProcessData(ReviewDataPath='./Data/Review/',
# 					MetaDataPath='./Data/Meta/',
# 					ReviewDataSavepath='./AfterPreprocessData/ReviewBin/',  
#				    Review_UserDataSavepath = './AfterPreprocessData/ReviewUserBin/',
#				    Review_CombineUserDataSavepath = './AfterPreprocessData/ReviewUserCombineBin/',
#				    MetaDataSavepath ='./AfterPreprocessData/MetaBin/', 
#				    Meta_CombineDataSavepath = './AfterPreprocessData/MetaCombineBin/', 
#				    ProcessInfoPath= './InfoData/')