import time,os,pickle,re
import numpy as np
import pandas as pd
from Data_loader.Auction import AuctionData
from Data_loader.Data_Util import ReadFileList

def ReadRawDataFile(Filename, Filepath, FileSavePath):
    BidderRecordDatas = pd.read_csv(Filepath, sep=';', engine='python')

    # Save Data
    pdsavepath = FileSavePath + Filename + '_Review_Bin.bin'
    with open(pdsavepath, 'wb+') as f:
        pickle.dump(BidderRecordDatas, f)
    print('BidderRecordDatas:',BidderRecordDatas)
    return BidderRecordDatas



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

def DoneAllFile(BidDataFilePath, BidDataFileProcessInfoPath, BidDataBinSavepath, BidUserBinSavePath):
    
    print("Start Review Data Process!\n")
    if not os.path.exists(BidDataFileProcessInfoPath):
        os.makedirs(BidDataFileProcessInfoPath)

    if not os.path.exists(BidDataBinSavepath):
        os.makedirs(BidDataBinSavepath)

    if not os.path.exists(BidUserBinSavePath):
        os.makedirs(BidUserBinSavePath)
    
    FileList = ReadFileList(BidDataFilePath)
    print('FileList in Review read: ', FileList)

    Filename_MaxBidLen_Dict = dict()
    
    for File in FileList:
        Filename = File.replace('reviews_','').replace('_5.json.gz','')
        InfoFilePath = BidDataFileProcessInfoPath + Filename + "_ReviewDataProcessInfo.txt"
        
        if os.path.exists(InfoFilePath):
            # print(Filename + " have been done!\n")
            continue

        # read data and write the program log
        with open(InfoFilePath, "w") as f:
            #print(Filename, "Start to read review data")
            startreadtime = time.time()
            BidDatas = ReadRawDataFile(Filename, BidDataFilePath + File, BidDataBinSavepath)
            endreadtime = time.time()
            #print(Filename, "end to read review data, time:", endreadtime - startreadtime)
            f.write("read review data time: %s s\n" % (str(endreadtime - startreadtime)))
            #reviewerID  asin reviewText  unixReviewTime  reviewTime

            #print(Filename, "Start to sort review data")
            startsorttime = time.time()
            BidDatas = BidDatas.sort_values(by=['asin','bid'], ascending=True) # 竞拍价格从小到大
            endsorttime = time.time()
            #print(Filename, "End to sort review data, time:", endsorttime - startsorttime)
            
            f.write("sort review data time: %s s\n" % (str(endsorttime - startsorttime)))

            #print(Filename, "Start to Aggregate User")
            startaggregatetime = time.time()
            UserList = itemAggregation(BidDatas)
            Filename_MaxBidLen_Dict[Filename] = int(GetMaxBidLen(UserList))
            endaggregatetime = time.time()
            #print(Filename, "End to Aggregate User, time:", endaggregatetime - startaggregatetime)
            f.write("Aggregate Use time: %s s\n" % (str(endaggregatetime - startaggregatetime)))


            with open(BidUserBinSavePath + Filename + '_User_Bin.bin', 'wb+') as ff:
                pickle.dump(UserList, ff)
            #print(Filename, "Done!")
    
    MaxBidLenFilePath = BidDataFileProcessInfoPath + "MaxBidLen.txt"
    if os.path.exists(MaxBidLenFilePath):
        with open(MaxBidLenFilePath, "r+") as fff:
            for line in fff.readlines():
                line = line.strip()
                k = line.split(':')[0]
                v = line.split(':')[1]
                Filename_MaxBidLen_Dict[k] = int(v)
    else:
        # record Each DataSet Max Bid Len
        with open(MaxBidLenFilePath, "w+") as fff:
            for k,v in Filename_MaxBidLen_Dict.items():
                FMInfo = str(k) + ":" + str(v) + "\n"
                fff.write(FMInfo)
    
    print("End Review Data Process!\n")
    print('Filename_MaxBidLen_Dict:', Filename_MaxBidLen_Dict)
    return Filename_MaxBidLen_Dict


# Test
#if __name__ == "__main__":
#    DoneAllFile(BidDataFilePath = '../Data/AmazonData/review/Part/' , BidDataFileProcessInfoPath = './InfoData/', BidDataBinSavepath = './AfterPreprocessData/ReviewBin/', BidUserBinSavePath = './AfterPreprocessData/ReviewUserBin/')
