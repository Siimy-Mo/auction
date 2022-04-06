import Data_loader.Data_Generator_auction as data
import Model.SharedEmbedding as SE
import Model.model_test as model_test
import os,json,argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def train(psammodel, Embed, Dataset, params):
    filename = "./Performance_PSAM_Model_Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_lr_" + str(params.learning_rate) + "_embsize_" + str(params.embed_size) + "_numunit_" + str(params.num_units) + ".txt"
    filename_epoch = "./Performance_PSAM_Model_Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_lr_" + str(params.learning_rate) + "_embsize_" + str(params.embed_size) + "_numunit_" + str(params.num_units) + "_each_Epoch" +".txt"
    model_dir = params.model + "Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_embsize_" + str(params.embed_size) + "_numunit_" + str(params.num_units) +"/"
    #pig_dir = params.pig + "Windowsize_" + str(params.window_size) + "_Epoch_" +  str(params.epoch) + "_embsize_" + str(params.embed_size)  + "_numunit_" + str(params.num_units) +"/"
    
    print('Train func filename: ', filename)
    print('Train func model_dir', model_dir)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    #if not os.path.isdir(pig_dir):
        #os.mkdir(pig_dir)    
    log_dir = os.path.join(model_dir, 'logs')
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session:

        # train dataset
        train_dataset = np.array(Dataset.train_data)
        np.random.shuffle(train_dataset)
        
        # eval dataset
        # eval_dataset = np.array(Dataset.eval_data)
        # np.random.shuffle(eval_dataset)
        
        # train_dataset = np.concatenate((train_dataset,eval_dataset),axis=0)

        usernum = Dataset.userNum
        auctionnum = Dataset.auctionNum
        bidnum = Dataset.bidNum

        print(" Users/Auction/Bid:", usernum[0], auctionnum[0], bidnum[0])
        
        avg_loss = tf.compat.v1.placeholder(tf.float32, [], 'loss')
        tf.compat.v1.summary.scalar('loss', avg_loss)

        validation_HR = tf.compat.v1.placeholder(tf.float32, [], 'validation_HR')
        tf.compat.v1.summary.scalar('validation_HR', validation_HR)
        
        validation_MRR = tf.compat.v1.placeholder(tf.float32, [], 'validation_MRR')
        tf.compat.v1.summary.scalar('validation_MRR', validation_MRR)
        
        validation_NDCG = tf.compat.v1.placeholder(tf.float32, [], 'validation_NDCG')
        tf.compat.v1.summary.scalar('validation_NDCG', validation_NDCG)
        

        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)
        summaries = tf.compat.v1.summary.merge_all()
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        train_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        
        session.run(tf.compat.v1.local_variables_initializer())
        session.run(tf.compat.v1.global_variables_initializer())

        # 画一个Bid系数统计图~ 【大部分数据位于[1,23]】
        # distributionPlt(Dataset.bidCoef,'bidCoef')
        # distributionPlt(Dataset.timeintervalList,'timeDiff')
        # distributionPlt(Dataset.timestampList,'timeStamp')

        # define loss
        losses = []
        pos_losses = []
        neg_losses = []

        # define performance indicator
        HR10_list,HR20_list,HR50_list = [],[],[]
        MRR10_list,MRR20_list,MRR50_list = [],[],[]
        NDCG10_list,NDCG20_list,NDCG50_list = [],[],[]
        allSets_auction_list, auction_testOnly_list = [],[]
        opt_loss_list=[]

        total_batch = int(len(train_dataset) / params.batch_size) + 1
        step = 0
        min_loss = 10000.
        best_val = 10000. 
        best_HR10,best_HR20,best_HR50 = 0., 0., 0.
        best_MRR10,best_MRR20,best_MRR50 = 0., 0., 0.
        best_NDCG10,best_NDCG20,best_NDCG50 = 0., 0., 0.

        test_dataset = Dataset.test_data
        test_auction_list_4d= Dataset.testAuction_attrs    # 813
        test100_list = []
        test_auction_list = [x for x in test_auction_list_4d]
        for i in range(len(test_dataset)):# 64 times
            pos_item = test_dataset[i][-4]
            withoutTest = []
            for index in range(len(test_auction_list)):
                if pos_item[0] != test_auction_list[index][0]:
                    withoutTest.append(test_auction_list[index])
            test_list = random.sample(withoutTest,99)+[pos_item]
            test100_list.append(test_list)
        # test100_list = [test_auction_list_4d[x] for x in test100_id_list]
        test100_list = np.array(test100_list)# 还需要attr的array

        for e in range(params.epoch):
            train_startpos = 0
            for b in range(total_batch):
                u_train,bpp_train,sl_train,btf_train,lbpp_train,lbppl_train,lbtp_train,cpp_train,cni_train,cptp_train,cpbl_train \
                    = data.Get_next_batch(Dataset, train_dataset, train_startpos, params.batch_size)
                # print(u_train.shape,bpp_train.shape,sl_train.shape,bpf_train.shape,cpp_train.shape,cni_train.shape,cbpd_train.shape,cbpo_train.shape,bpd_train.shape,bpo_train.shape)
                _, train_loss, train_pos_loss, train_neg_loss = psammodel.step(session, u_train,bpp_train,sl_train,btf_train,lbpp_train,lbppl_train,lbtp_train,cpp_train,cni_train,cptp_train,cpbl_train)
                
                losses.append(train_loss)
                pos_losses.append(train_pos_loss)
                neg_losses.append(train_neg_loss)
                train_startpos += params.batch_size
                step += 1
                #print(train_startpos)
                #if step % params.log_every == 0 and step != 0:
                    #print("step: {}|  epoch: {}|  batch: {}|  train_loss: {:.6f}| pos_loss: {:.6f} | neg_loss: {:.6f}".format(step, e, b, train_loss, train_pos_loss, train_neg_loss))
                
            print("step: {}|  epoch: {}|  batch: {}|  train_loss: {:.6f}| pos_loss: {:.6f} | neg_loss: {:.6f}".format(step, e, b, train_loss, train_pos_loss, train_neg_loss))
            
            # After every epoch
            #test performance
            test_dataset = np.array(Dataset.test_data)
            # test_auction_list_4d= Dataset.testAuction_attrs    # 813
            u_test,bpp_test,sl_test,btf_test,lbpp_test,lbppl_test,lbtp_test,cpp_test,cni_test,cpt_test,cpbl_test  \
                = data.Get_next_batch(Dataset, test_dataset, 0, len(test_dataset))

            #print("Start Performance Test of point model!\n")
            # test100_list = []
            # test_auction_list = [x for x in test_auction_list_4d]
            # for i in range(len(cpp_test)):# 64 times
            #     pos_item = cpp_test[i].tolist()
            #     withoutTest = []
            #     for index in range(len(test_auction_list)):
            #         if pos_item[0] != test_auction_list[index][0]:
            #             withoutTest.append(test_auction_list[index])
            #     test_list = random.sample(withoutTest,99)+[pos_item]
            #     test100_list.append(test_list)
            # # test100_list = [test_auction_list_4d[x] for x in test100_id_list]
            # test100_list = np.array(test100_list)# 还需要attr的array

            bs, itememb, allproductemb,opt_loss = psammodel.step(session, u_test,bpp_test,sl_test,btf_test,lbpp_test,lbppl_test,lbtp_test,cpp_test,cni_test,cpt_test,cpbl_test,test100_list,True)
            all_hr_10,all_mrr_10,all_ndcg_10 = 0,0,0
            all_hr_20,all_mrr_20,all_ndcg_20 = 0,0,0
            all_hr_50,all_mrr_50,all_ndcg_50 = 0,0,0
            repeatSet = set()
            allSets_auction_count,auction_testOnly_count=0,0
            for i in range(len(itememb)):
                testTarget = str(u_test[i]) + '_' + str(cpp_test[i][0])
                repeatNewSet = set.union(repeatSet, {testTarget})
                if len(repeatNewSet) == len(repeatSet): pass
                else:
                    repeatSet.add(testTarget)

                    # testid： cpp_test[i] test_auction_list
                    RankList_10, per_hr_10,per_mrr_10,per_ndcg_10,per_hr_20,per_mrr_20,per_ndcg_20,per_hr_50,per_mrr_50,per_ndcg_50  \
                         = me.GetItemRankList(allproductemb[i,:], test100_list[i][:,0], itememb[i], cpp_test[i][0])

                    all_hr_10 += per_hr_10
                    all_hr_20 += per_hr_20
                    all_hr_50 += per_hr_50
                    all_mrr_10+=per_mrr_10
                    all_mrr_20+=per_mrr_20
                    all_mrr_50+=per_mrr_50
                    all_ndcg_10+=per_ndcg_10
                    all_ndcg_20+=per_ndcg_20
                    all_ndcg_50+=per_ndcg_50

                    # 分开追踪Hit
                    allSets_auction = Dataset.predictionReport(u_test[i],cpp_test[i], RankList_10,e)
                    allSets_auction_count += allSets_auction
            hr10 = all_hr_10 / float(len(repeatSet))
            mrr10 = all_mrr_10 / float(len(repeatSet))
            ndcg10 = all_ndcg_10 / float(len(repeatSet))
            hr20 = all_hr_20 / float(len(repeatSet))
            mrr20 = all_mrr_20 / float(len(repeatSet))
            ndcg20 = all_ndcg_20 / float(len(repeatSet))
            hr50 = all_hr_50 / float(len(repeatSet))
            mrr50 = all_mrr_50 / float(len(repeatSet))
            ndcg50 = all_ndcg_50 / float(len(repeatSet))

            Performance_info = "Performance: EPOCH:{}|HR10: {:.4f}|HR20: {:.4f}|HR50: {:.4f}| MRR10: {:.4f}|MRR20: {:.4f}|MRR50: {:.4f}|NDCG10: {:.4f}|NDCG20: {:.4f}|NDCG50: {:.4f}".format(e, hr10,hr20,hr50, mrr10,mrr20,mrr50, ndcg10,ndcg20,ndcg50)


            print(Performance_info)
            with open(filename_epoch, 'a+') as f:
                f.write(Performance_info+'\n')
            # update the best performance 暂时只显示top10
            HR10_list.append(hr10)
            MRR10_list.append(mrr10)
            NDCG10_list.append(ndcg10)
            HR20_list.append(hr20)
            MRR20_list.append(mrr20)
            NDCG20_list.append(ndcg20)
            HR50_list.append(hr50)
            MRR50_list.append(mrr50)
            NDCG50_list.append(ndcg50)
            allSets_auction_list.append(allSets_auction_count)
            auction_testOnly_list.append(auction_testOnly_count)
            opt_loss_list.append(opt_loss)
            print('预测中了：',allSets_auction_count,'Total(user-item):',len(repeatSet))

            can_save = False
            if best_HR10 < hr10:
                best_HR10 = hr10
                best_HR20 = hr20
                best_HR50 = hr50
                can_save = True
            if best_MRR10 < mrr10:
                best_MRR10 = mrr10
                best_MRR20 = mrr20
                best_MRR50 = mrr50
                can_save = True
            if best_NDCG10 < ndcg10:
                best_NDCG10 = ndcg10
                best_NDCG20 = ndcg20
                best_NDCG50 = ndcg50
                can_save = True
            if can_save == True:
                saver.save(session, model_dir, global_step=step)

        with open(filename, 'a+') as f:
            bestinfo = "Best Performance: HR10: {:.4f}|HR20: {:.4f}|HR50: {:.4f}| MRR10: {:.4f}|MRR20: {:.4f}|MRR50: {:.4f}|NDCG10: {:.4f}|NDCG20: {:.4f}|NDCG50: {:.4f}".format(best_HR10,best_HR20,best_HR50, best_MRR10,best_MRR20,best_MRR50, best_NDCG10,best_NDCG20,best_NDCG50)
            f.write(bestinfo+'\n')
    # for i in range(len(u_test)):
    #     if hitList.count(u_test[i]) >0:
    #         print('The user',u_test[i],'for auction ',cpp_test[i],'appears times: ',hitList.count(u_test[i]))
    x_data = range(params.epoch)
    l9=plt.plot(x_data,opt_loss_list,'b--',label='opt_loss_list')
    plt.plot(x_data,opt_loss_list,'ro-')
    plt.title('The opt_loss metrics in Three Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    l1=plt.plot(x_data,HR10_list,'r--',label='HR10_list')
    l2=plt.plot(x_data,HR20_list,'g--',label='HR20_list')
    l3=plt.plot(x_data,HR50_list,'b--',label='HR50_list')
    plt.plot(x_data,HR10_list,'ro-',x_data,HR20_list,'g+-',x_data,HR50_list,'b^-')
    plt.title('The HR metrics in Three Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()
    input()

    l4=plt.plot(x_data,MRR10_list,'r--',label='MRR10_list')
    l5=plt.plot(x_data,MRR20_list,'g--',label='MRR20_list')
    l6=plt.plot(x_data,MRR50_list,'b--',label='MRR50_list')
    plt.plot(x_data,MRR10_list,'ro-',x_data,MRR20_list,'g+-',x_data,MRR50_list,'b^-')
    plt.title('The MRR metrics in Three Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()
    input()

    l7=plt.plot(x_data,NDCG10_list,'r--',label='NDCG10_list')
    l8=plt.plot(x_data,NDCG20_list,'g--',label='NDCG20_list')
    l9=plt.plot(x_data,NDCG50_list,'b--',label='NDCG50_list')
    plt.plot(x_data,NDCG10_list,'ro-',x_data,NDCG20_list,'g+-',x_data,NDCG50_list,'b^-')
    plt.title('The NDCG metrics in Three Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()

    input()

def experimentsInfoReport(params):
    info0 = 'Experiments Information (User number: {0}): \n'.format(params['totoal User number'])
    timelines = [params['eval timeline'] - params['MinMax Unix time'][0], params['test timeline'] - params['MinMax Unix time'][0] ]
    trainUser = params['train number'][0]
    trainAuction = params['train number'][1]
    testUser = params['test number'][0]
    testAuction = params['test number'][1]
    info1 = 'Timelines of eval and test are {0} and {1} with coeffs {2} and {3}\n'.format(timelines[0], timelines[1],params['Timeline coeffs'][0],params['Timeline coeffs'][1])
    info1 = 'Timelines of eval and test are {0} and {1} with coeffs {2} and {3}\n'.format(timelines[0], timelines[1],params['Timeline coeffs'][0],params['Timeline coeffs'][1])
    info2 = 'Numbers of dataset are {0}, {1} (Users); {2}, {3}(Auctions) \n'.format( trainUser, testUser, trainAuction, testAuction)
    info3 = 'Proportions(Auction/User): {:.2f} (Train) and {:.2f} (Test) \n'.format( trainUser/trainAuction, testUser/testAuction)
    info4 = 'Test auction number in both periods: {0}, in Test period Only: {1}\n'.format( params['auction number in both'], params['auction number in test only'])

    return info0 + info1 + info2 + info3 + info4

def distributionPlt(datalist, type):
    # bid coefficient:
    if type=='bidCoef':
        bidcoef = []
        for i in datalist:
            bidcoef.append(datalist[i])
            if datalist[i] > 40.0:
                print(datalist[i])
        plt.hist(bidcoef,normed=0,facecolor="blue",edgecolor="black",alpha=0.7)
        plt.title('The bid Coefficient Distribution')
    elif type=='timeDiff':
        plt.hist(datalist,normed=0,facecolor="blue",edgecolor="black",alpha=0.7)
        plt.title('The time interval Distribution')
    elif type=='timeStamp':
        plt.hist(datalist,normed=0,facecolor="blue",edgecolor="black",alpha=0.7)
        plt.title('The timestamp Distribution')
    plt.xlabel('row')
    plt.ylabel('column')
    plt.legend()
    plt.show()

def main(args):
    if not os.path.isdir(args.model):
        os.mkdir(args.model)

    with open(os.path.join(args.model, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    Dataset = data.Generate_Data(args) # 根據時間分類，在Data中，修改DataSet Class
    Embed = SE.Embedding(Dataset, args)
    psammodel = model_test.Seq(Embed, args)
    # MF = MF_model.MF_layer(Embed, args)
    train(psammodel, Embed, Dataset, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--pig', type=str, default="./picture/",
                        help='path to picture output directory')
    parser.add_argument('--ReviewDataPath', type=str, required=True,
    					help='path to the input review dataset')
    parser.add_argument('--MetaDataPath', type=str, required=True,
    					help='path to the input meta dataset')

    # set default is fine.
    parser.add_argument('--ReviewDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewBin/',
    					help='path to the save the bin of review dataset')
    parser.add_argument('--Review_UserDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewUserBin/',
    					help='path to the save the bin of user Review dataset')
    parser.add_argument('--Review_CombineUserDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/ReviewUserCombineBin/',
    					help='path to the save the bin of Review dataset combined by user')
    parser.add_argument('--MetaDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/MetaBin/',
    					help='path to the save the bin of Meta dataset')
    parser.add_argument('--Meta_CombineDataSavepath', type=str, default='./Data_loader/AfterPreprocessData/MetaCombineBin/',
    					help='path to the save the combine bin of Meta dataset')
    parser.add_argument('--ProcessInfoPath', type=str, default='./Data_loader/InfoData/',
    					help='path to the save the Info of process data')
    parser.add_argument('--DataSetSavePath', type=str, default='./Data_loader/AfterPreprocessData/DataSet/',
    					help='path to the save the Info of process data')
    

    parser.add_argument('--user-emb', type=str, default="Short_term",
                        help='Select type of user embedding: Complete, Short_term and Long_term')
    parser.add_argument('--loss-f', type=str, default="Inner_product",
                        help='Select type of loss function: MetricLearning and Inner_product')
    parser.add_argument('--window-size', type=int, default=5,
                        help='ProNADE Input Data Window size(0:Use all product bought before)')
    parser.add_argument('--embed-size', type=int, default=128,
                        help='size of the hidden layer of User, product and query')
    parser.add_argument('--short-term-size', type=int, default=5,
                        help='size of User short-term preference')
    parser.add_argument('--long-term-size', type=int, default=15,
                        help='size of User lhort-term preference')                
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='which activation to use: sigmoid|tanh')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=200,
                        help='train data epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--validation-batch-size', type=int, default=64,
                        help='the batch size of validation')              
    parser.add_argument('--neg-sample-num', type=int, default=5, # 原来5
                        help='the number of negative sample')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--depth', type=int, default=50,
                        help='the depth of test performance to use')
    parser.add_argument('--modelsel', type=int, default=1,
                        help='model selection')

    # LSTM-Parameter
    parser.add_argument('--num-units', type=int, default=128,
                        help='the number of hidden unit in lstm model')                 
    
    # Transformer-Parameter
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    return parser.parse_args()

    
if __name__ == "__main__":
    main(parse_args())