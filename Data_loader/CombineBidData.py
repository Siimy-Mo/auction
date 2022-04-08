import operator,glob,pickle,os
from Data_loader.Auction import AuctionData
from Data_loader.Data_Util import ReadFileList

def FindTheSameUser(U1, U2):
    User1List = []
    for i in range(len(U1)):
        User1List.append(U1[i].UserID)
    User2List = []
    for i in range(len(U2)):
        User2List.append(U2[i].UserID)
    Intersection_User = [x for x in User1List if x in User2List]
    CombineList = []
    for i in range(len(U1)):
        if U1[i].UserID in Intersection_User:
            CombineList.append(U1[i])
    for i in range(len(U2)):
        if U2[i].UserID in Intersection_User:
            CombineList.append(U2[i])
    
    cmpfun = operator.attrgetter('UserID') # 用来替代只为了获取attr的lambda函数。對象是class，如對象是dict的話用itemgetter
    CombineList.sort(key=cmpfun)
    return CombineList

def CombineUser(U1, U2):
    CombineList = []
    for i in range(len(U1)):
        CombineList.append(U1[i])
    for i in range(len(U2)):
        CombineList.append(U2[i])
    cmpfun = operator.attrgetter('UserID')
    CombineList.sort(key=cmpfun)
    return CombineList

def Getmaxproductnum(AllBidUserList):
    UserProLen = []
    for i in range(len(AllBidUserList)):
        eachUserProlen = AllBidUserList[i].GetUserBidLen()
        UserProLen.append(eachUserProlen)
    MaxProductNum = max(UserProLen)
    return MaxProductNum

def CombineBidDataSetByUser(BidUserBinSavePath, CombineBidUserBinProcessPath, CombineBidUserBinPath):
    print(BidUserBinSavePath, CombineBidUserBinProcessPath, CombineBidUserBinPath)

    if not os.path.exists(CombineBidUserBinPath):
        os.makedirs(CombineBidUserBinPath)

    BidUserFileList = ReadFileList(BidUserBinSavePath)
    
    BidUserCombineSaveFilePath = CombineBidUserBinPath
    
    AllBidUserList = []
    FileNameList = []
    
    for i in range(len(BidUserFileList)):
        EachFileName = BidUserFileList[i].replace('_User_Bin.bin','')
        FileNameList.append(EachFileName)
    for i in range(len(FileNameList)):
        BidUserCombineSaveFilePath = BidUserCombineSaveFilePath + FileNameList[i] + "_"
    BidUserCombineSaveFilePath = BidUserCombineSaveFilePath + '_Combine_User_Bin.bin'
    if os.path.exists(BidUserCombineSaveFilePath):
        print("Load Combine Bid Data!\n")
        with open(BidUserCombineSaveFilePath, "rb+") as f:
            AllBidUserList = pickle.load(f)
        MaxProductNum = Getmaxproductnum(AllBidUserList)
        print("Load Combine Bid Data Finished!\n")
        return AllBidUserList, MaxProductNum
    else:
        print("Start Combine Bid Data!\n")
        FileUserNum = []
        for i in range(len(BidUserFileList)):
            with open(BidUserBinSavePath + BidUserFileList[i], 'rb+') as f:
                EachBidUser = pickle.load(f)
                FileUserNum.append(len(EachBidUser))
                AllBidUserList.append(EachBidUser)
        #print("Load Bid User File Done!")
        TimeToOperate = len(AllBidUserList) - 1
        for t in range(TimeToOperate):
            I_UserList = AllBidUserList[0]
            J_UserList = AllBidUserList[1]
            del AllBidUserList[0:2]
            I_J_UserList = CombineUser(I_UserList, J_UserList)
            CombineUserList  = []
            CombineUserList.append(I_J_UserList[0])
            for k in range(1, len(I_J_UserList)):
                if I_J_UserList[k].UserID == CombineUserList[len(CombineUserList) - 1].UserID:
                    for l in range(len(I_J_UserList[k].UserPurchaseList)):
                        CombineUserList[len(CombineUserList) - 1].AddPurchase(I_J_UserList[k].UserPurchaseList[l])
                else:
                    CombineUserList[len(CombineUserList) - 1].UserPurchaseList = sorted(CombineUserList[len(CombineUserList) - 1].UserPurchaseList, key=operator.itemgetter('unixBidTime'))
                    CombineUserList.append(I_J_UserList[k])
            AllBidUserList.append(CombineUserList)
        
        # 记录用户的競拍序列的最长长度
        # -> 记录一场拍卖会的竞拍序列最长长度
        MaxProductNum = Getmaxproductnum(AllBidUserList[0])
        
        RUCInfo = ''
        
        for i in range(len(FileNameList)):
            RUCInfo = RUCInfo + FileNameList[i] + ' has user: ' + str(FileUserNum[i]) + "\n"
        RUCInfo = RUCInfo + "CombineList has user: " + str(len(AllBidUserList[0])) + "\n"
        with open(CombineBidUserBinProcessPath + "BidUserCombineInfo.txt", "w+") as f:
            f.write(RUCInfo)
        with open(BidUserCombineSaveFilePath, "wb+") as f:
            pickle.dump(AllBidUserList[0], f)
        print('End Combine Bid Data!\n')
        return AllBidUserList[0], MaxProductNum
    
# Test
#if __name__ == "__main__":
#    CombineBidDataSetByUser(BidUserBinSavePath='./AfterPreprocessData/BidUserBin/', CombineBidUserBinProcessPath='./InfoData/',CombineBidUserBinPath='./AfterPreprocessData/BidUserCombineBin/')
