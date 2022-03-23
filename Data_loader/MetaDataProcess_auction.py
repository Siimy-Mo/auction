import gzip,time,pickle,os
import pandas as pd
from Data_loader.Data_Util import ReadFileList

def ReadDataFile(data_path):
    # asin;name;openbid;auction_duration;unixEndDate;endDate;bidders;bids
    ReviewDatas = pd.read_csv(data_path, sep=';', engine='python')
    values = []
    for index, row in ReviewDatas.iterrows():
        titleList=row['name'].split(' ')
        # asin;type;name;openbid;price;auction_duration;unixEndDate;endDate;bidders;bids;test
        values.append({'asin':row['asin'],'type':row['type'],'name': row['name'],'openbid':row['openbid'],'price':row['price'],
                        'auction_duration':row['auction_duration'],'unixEndDate':row['unixEndDate'],'endDate':row['endDate'],'bidders':row['bidders'],
                        'bids':row['bids'],'nameList':titleList})

    return values


def get_query(x):
    qs = list()
    for sub_cat_list in x:
        if (len(sub_cat_list) <= 1):
            continue
        qs.append(sub_cat_list)
   
    finalQs = []
    
    for q in qs:
        
        Q_words = ' '.join(q).lower().replace(' & ', ' ').replace(',', '').strip().split(' ')
        finalQ = ''
        words = set()
        for i in range(len(Q_words)-1, -1, -1):
            if (Q_words[i] not in words):
                finalQ = Q_words[i] + ' ' + finalQ
                words.add(Q_words[i])
        finalQs.append( finalQ.strip())
    return finalQs

# 记录最大的query长度
def GetMaxLength(querylist):
    q_lens = []
    for i in querylist:
        for q in i:
            q_lens.append(len(q.split(' ')))
    max_query_len = max(q_lens)
    return max_query_len

def DoneAllFile(MetaDataFilePath, MetaDataFileProcessInfopath, MetaDataBinSavePath):
    print("Start Meta Data Process!\n")
    if not os.path.exists(MetaDataFileProcessInfopath):
        os.makedirs(MetaDataFileProcessInfopath)

    if not os.path.exists(MetaDataBinSavePath):
        os.makedirs(MetaDataBinSavePath)


    MetaDataFileList = ReadFileList(MetaDataFilePath)
    Filename_MaxNameLen = dict()
    for i in range(len(MetaDataFileList)):
        MetaFileName = MetaDataFileList[i].replace('meta_','').replace('.json.gz','')
        InfoFilePath = MetaDataFileProcessInfopath + MetaFileName + "_MetaDataProcessInfo.txt"
        if os.path.exists(InfoFilePath):
            #print(MetaFileName + " have been done!\n")
            continue
        with open(InfoFilePath, "w+") as f:
            #print(MetaFileName, "Start to read meta data")
            startreadtime = time.time()
            meta_datas = pd.DataFrame(ReadDataFile(MetaDataFilePath + MetaDataFileList[i]))
            endreadtime = time.time()
            #print(MetaFileName, "end to read review data, time:", endreadtime - startreadtime)
            f.write("read auction data time: %s s\n" % (str(endreadtime - startreadtime)))

            meta_datas.set_index('asin', inplace=True)
            #print(MetaFileName, "Start to get query")
            startgetquerytime = time.time()
            meta_datas['nameList'] = meta_datas['nameList'].map(get_query)  # 統一化：Apple, -> a p p l e,
            # Get Max Query Len
            eachMQL = GetMaxLength(meta_datas['nameList'])
            Filename_MaxNameLen[MetaFileName] = eachMQL
            endgetquerytime = time.time()
            #print(MetaFileName, "end to get query, time:", endgetquerytime - startgetquerytime)
            f.write("get name time: %s s\n" % (str(endgetquerytime - startgetquerytime)))
            
            with open(MetaDataBinSavePath + MetaFileName + '_Meta_Bin.bin', 'wb+') as ff:
                pickle.dump(meta_datas, ff)
            #print(MetaFileName, "Done!")
    
    MaxNameLenFilePath = MetaDataFileProcessInfopath + "MaxNameLen.txt"
    if os.path.exists(MaxNameLenFilePath):
        with open(MaxNameLenFilePath, "r+") as fff:
            for line in fff.readlines():
                line = line.strip()
                k = line.split(':')[0]
                v = line.split(':')[1]
                Filename_MaxNameLen[k] = int(v)
    else:
        with open(MaxNameLenFilePath, "w+") as fff:
            for k,v in Filename_MaxNameLen.items():
                FMInfo = str(k) + ":" + str(v) + "\n"
                fff.write(FMInfo)
    
    print("End Meta Data Process!\n")
    return Filename_MaxNameLen

# Test
#if __name__ == "__main__":
    #DoneAllFile(MetaDataFilePath='../Data/AmazonData/metadata/', MetaDataFileProcessInfopath='./InfoData/',MetaDataBinSavePath='./AfterPreprocessData/MetaBin/')