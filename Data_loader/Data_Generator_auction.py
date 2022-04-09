import os,pickle,errno
import Data_loader.Data_Util as Data_Util
import Data_loader.BidRecordDataProcess as BidRecordDataProcess
import Data_loader.CombineBidData as CombineBidData
import Data_loader.MetaDataProcess_auction as MetaDataProcess
import Data_loader.CombineMetaData as CombineMetaData
import Data_loader.Data_auction as Data
import tensorflow as tf
import numpy as np
import random


def PreProcessData(params):
	print("Start PreProcess Dataset!\n")
	# Read the review dataset
	BidFilename_MaxBidLen_Dict = BidRecordDataProcess.DoneAllFile(BidDataFilePath=params.ReviewDataPath, BidDataFileProcessInfoPath=params.ProcessInfoPath, BidDataBinSavepath=params.ReviewDataSavepath, BidUserBinSavePath=params.Review_UserDataSavepath)

	# # Combine the User from all the review dataset, 這裡用了bid，BidderData中還沒有追加Buy的product
	BidAuctionCombineData,max_auction_bids_len = CombineBidData.CombineBidDataSetByUser(BidUserBinSavePath=params.Review_UserDataSavepath, CombineBidUserBinProcessPath=params.ProcessInfoPath,CombineBidUserBinPath=params.Review_CombineUserDataSavepath)

	# # Read the meta dataset
	MetaFilename_MaxItemLen_Dict = MetaDataProcess.DoneAllFile(MetaDataFilePath=params.MetaDataPath, MetaDataFileProcessInfopath=params.ProcessInfoPath,MetaDataBinSavePath=params.MetaDataSavepath)

	# # Combine all the meta dataset
	MetaCombineData = CombineMetaData.CombineMetaDataSet(MetaDataBinSavePath=params.MetaDataSavepath, CombineMetaDataBinProcessInfoPath=params.ProcessInfoPath,CombineMetaDataBinSavePath=params.Meta_CombineDataSavepath)

	# max_bid_len = max(zip(BidFilename_MaxBidLen_Dict.values(), BidFilename_MaxBidLen_Dict.keys()))[0] 
	# max_name_len = max(zip(MetaFilename_MaxItemLen_Dict.values(), MetaFilename_MaxItemLen_Dict.keys()))[0] 

	# Create dataset
	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
	print('DataSetSavePath',DataSetSavePath)

	# 1是dict，2是df
	print(params.neg_sample_num, max_auction_bids_len, DataSetSavePath, params.short_term_size, params.long_term_size)
	Dataset = Data.DataSet(BidAuctionCombineData,MetaCombineData, params.neg_sample_num, max_auction_bids_len, params.short_term_size, params.long_term_size)


	print("End PreProcess Dataset!\n")
	return Dataset

def LoadData(params, DataSetSavePath, SaveFile):
	print("Start Load Dataset!\n")
	DataSetSavePath = DataSetSavePath + SaveFile
	with open(DataSetSavePath, "rb+") as f:
		Dataset = pickle.load(f)
	
	print("End Load Dataset!\n")
	
	#train_dataset = tf.data.Dataset.from_generator(Dataset.next_train_batch, (tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32))
	# need repeat epoch and set batch_size
	#train_dataset = train_dataset.repeat(params.epoch).batch(params.batch_size)
	return Dataset


def Generate_Data(params):
	if not os.path.exists(params.ReviewDataPath):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params.ReviewDataPath)

	if not os.path.exists(params.MetaDataPath):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params.MetaDataPath)

	# CategorySet = list(Data_Util.GetCategory(params.MetaDataPath))
	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
	if not os.path.exists(DataSetSavePath):
		os.makedirs(DataSetSavePath)
	
	# SaveFile = Data_Util.FindFile(DataSetSavePath, CategorySet) 
	# if (SaveFile is None):
	# 	Dataset = PreProcessData(params, CategorySet)
	# else:
	# 	Dataset = LoadData(params, DataSetSavePath, SaveFile) 
	Dataset = PreProcessData(params)

	# use DataSetSavePath
	return Dataset
	# return None
	
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
		short_before_uid_pos = [dataset[i,3] for i in range(startpos, dataset_len)]
		short_before_uid_pos_len = [dataset[i,4] for i in range(startpos, dataset_len)]
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
		short_before_uid_pos = [dataset[i,3] for i in range(startpos, startpos + batch_size)]
		short_before_uid_pos_len = [dataset[i,4] for i in range(startpos, startpos + batch_size)]
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

	return np.array(pids), np.array(current_uid_pos), \
			np.array(short_before_uid_pos),np.array(short_before_uid_pos_len),np.array(CNIList),np.array(test_user_probs)

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