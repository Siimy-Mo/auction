import os,pickle,errno
import Data_loader.Data_Util as Data_Util
import Data_loader.BidRecordDataProcess as BidRecordDataProcess
import Data_loader.CombineBidData as CombineBidData
import Data_loader.MetaDataProcess_auction as MetaDataProcess
import Data_loader.CombineMetaData as CombineMetaData
import Data_loader.Data_auction as Data
import tensorflow as tf
import numpy as np


def PreProcessData(params):
	print("Start PreProcess Dataset!\n")
	# Read the review dataset
	BidFilename_MaxBidLen_Dict = BidRecordDataProcess.DoneAllFile(BidDataFilePath=params.ReviewDataPath, BidDataFileProcessInfoPath=params.ProcessInfoPath, BidDataBinSavepath=params.ReviewDataSavepath, BidUserBinSavePath=params.Review_UserDataSavepath)

	# # Combine the User from all the review dataset, 這裡用了bid，BidderData中還沒有追加Buy的product
	BidUserCombineData,max_product_len = CombineBidData.CombineBidDataSetByUser(BidUserBinSavePath=params.Review_UserDataSavepath, CombineBidUserBinProcessPath=params.ProcessInfoPath,CombineBidUserBinPath=params.Review_CombineUserDataSavepath)

	# # Read the meta dataset
	MetaFilename_MaxItemLen_Dict = MetaDataProcess.DoneAllFile(MetaDataFilePath=params.MetaDataPath, MetaDataFileProcessInfopath=params.ProcessInfoPath,MetaDataBinSavePath=params.MetaDataSavepath)

	# # Combine all the meta dataset
	MetaCombineData = CombineMetaData.CombineMetaDataSet(MetaDataBinSavePath=params.MetaDataSavepath, CombineMetaDataBinProcessInfoPath=params.ProcessInfoPath,CombineMetaDataBinSavePath=params.Meta_CombineDataSavepath)

	max_bid_len = max(zip(BidFilename_MaxBidLen_Dict.values(), BidFilename_MaxBidLen_Dict.keys()))[0] 
	max_name_len = max(zip(MetaFilename_MaxItemLen_Dict.values(), MetaFilename_MaxItemLen_Dict.keys()))[0] 

	# Create dataset
	DataSetSavePath = params.DataSetSavePath + "short_term_size_" + str(params.short_term_size) + "_long_term_size_" + str(params.long_term_size) + "/"
	print('DataSetSavePath',DataSetSavePath)

	# 1是dict，2是df
	print(params.neg_sample_num, max_name_len, max_bid_len, max_product_len, DataSetSavePath, params.short_term_size, params.long_term_size)
	Dataset = Data.DataSet(BidUserCombineData,MetaCombineData,params.neg_sample_num, max_name_len,max_bid_len,max_product_len, DataSetSavePath, params.short_term_size,params.long_term_size)

	# # for i in range(len(CategorySet)):
	# # 	DataSetSavePath = DataSetSavePath + CategorySet[i] + "_"
	# # with open(DataSetSavePath + ".bin", "wb+") as f:
	# # 	pickle.dump(Dataset, f)

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
	# ((uid,cur_before_pids_pos,cur_before_pids_pos_len, 
	# cur_before_pids_time_diff, cur_before_pids_type, \
                                        # current_pid_pos, current_pid_time, current_pid_type))
	dataset_len = len(dataset)
	CNIList = []
	if (startpos + batch_size) > dataset_len:
		uids = [dataset[i,0] for i in range(startpos, dataset_len)]
		before_pids_pos = [dataset[i,1] for i in range(startpos, dataset_len)]
		before_pids_pos_len = [dataset[i,2] for i in range(startpos, dataset_len)]
		current_before_pids_time_diff = [dataset[i,3] for i in range(startpos, dataset_len)]

		cur_long_before_pids_pos = [dataset[i,4] for i in range(startpos, dataset_len)]
		cur_long_before_pids_pos_len = [dataset[i,5] for i in range(startpos, dataset_len)]
		cur_long_before_time_pos = [dataset[i,6] for i in range(startpos, dataset_len)]

		current_pid_pos = [dataset[i,7] for i in range(startpos, dataset_len)]
		current_pid_time = [dataset[i,8] for i in range(startpos, dataset_len)]
		current_pid_bidderList = [dataset[i,9] for i in range(startpos, dataset_len)]
		for i in range(dataset_len - startpos):
			current_neg_item, current_neg_word = Dataset.neg_sample(current_pid_pos[i])
			CNIList.append(current_neg_item)
			if current_pid_pos[i] in CNIList[i]:
				print(current_pid_pos[i],CNIList[i])
			# CNWList.append(current_neg_word)
	else:
		uids = [dataset[i,0] for i in range(startpos, startpos + batch_size)]
		before_pids_pos = [dataset[i,1] for i in range(startpos, startpos + batch_size)]
		before_pids_pos_len = [dataset[i,2] for i in range(startpos, startpos + batch_size)]
		current_before_pids_time_diff = [dataset[i,3] for i in range(startpos, startpos + batch_size)]
		
		cur_long_before_pids_pos = [dataset[i,4] for i in range(startpos, startpos + batch_size)]
		cur_long_before_pids_pos_len = [dataset[i,5] for i in range(startpos, startpos + batch_size)]
		cur_long_before_time_pos = [dataset[i,6] for i in range(startpos, startpos + batch_size)]

		current_pid_pos = [dataset[i,7] for i in range(startpos, startpos + batch_size)]
		current_pid_time = [dataset[i,8] for i in range(startpos, startpos + batch_size)]
		current_pid_bidderList = [dataset[i,9] for i in range(startpos, startpos + batch_size)]
		for i in range(batch_size):
			current_neg_item, current_neg_word = Dataset.neg_sample(current_pid_pos[i])
			CNIList.append(current_neg_item)
			if current_pid_pos[i] in CNIList[i]:
				print(current_pid_pos[i],CNIList[i])
			# CNWList.append(current_neg_word)


	return np.array(uids), \
			np.array(before_pids_pos), np.array(before_pids_pos_len), np.array(current_before_pids_time_diff),\
			np.array(cur_long_before_pids_pos), np.array(cur_long_before_pids_pos_len), np.array(cur_long_before_time_pos),\
			np.array(current_pid_pos),np.array(CNIList), np.array(current_pid_time), np.array(current_pid_bidderList)
		#    np.array(current_pid_duration),np.array(current_pid_openbid),np.array(before_pids_duration),np.array(before_pids_openbid),\

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