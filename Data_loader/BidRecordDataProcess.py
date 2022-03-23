import gzip,json,time,os,pickle,nltk,re
import numpy as np
import pandas as pd
from Data_loader.Bidder import BidderData
from Data_loader.Data_Util import ReadFileList

def ReadRawDataFile(Filename, Filepath, FileSavePath):
    BidderRecordDatas = pd.read_csv(Filepath, sep=';', engine='python')

    # Save Data
    pdsavepath = FileSavePath + Filename + '_Review_Bin.bin'
    with open(pdsavepath, 'wb+') as f:
        pickle.dump(BidderRecordDatas, f)
    print('BidderRecordDatas:',BidderRecordDatas)
    return BidderRecordDatas

def help_f_cut_stop_word(x):
    x = x.lower()
    x = re.sub(r'([;\.~\!@\#\$\%\^\&\*\(\(\)_\+\=\-\[\]\)/\|\'\"\?<>,`\\])','',x)
    ss = ""
    words = x.split(' ')
    stopwords = nltk.corpus.stopwords.words('english') + list(';.~!@#$:%^&*(()_+=-[])/|\'\"?<>,`\\1234567890')
    for w in words:
        if (w in stopwords):
            pass
        else:
            ss += ' ' + w
    return ss.lower().strip()




def UserAggregation(BidData):
    UserList = []
    user_index = []
    for index, row in BidData.iterrows():
        uid = row['bidderID']
        if uid not in user_index:
            User = BidderData(row['bidderID'],row['bidderrate'])
            User.AddBid(row)
            UserList.append(User)
            user_index.append(uid)
        else:
            index = user_index.index(uid)
            UserList[index].AddBid(row)

    return UserList

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
    print(FileList)

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
            BidDatas = BidDatas.sort_values(by=['bidderID','unixBidTime'], ascending=True)
            endsorttime = time.time()
            #print(Filename, "End to sort review data, time:", endsorttime - startsorttime)
            
            f.write("sort review data time: %s s\n" % (str(endsorttime - startsorttime)))

            #print(Filename, "Start to Aggregate User")
            startaggregatetime = time.time()
            UserList = UserAggregation(BidDatas)
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
