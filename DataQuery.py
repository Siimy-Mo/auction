from numpy.core.defchararray import isdigit
import Data_loader.Data_Generator_auction as data
import Model.SharedEmbedding as SE
import Model.model_test as model_test
import os,json,argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt

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
        eval_dataset = np.array(Dataset.eval_data)
        np.random.shuffle(eval_dataset)
        
        train_dataset = np.concatenate((train_dataset,eval_dataset),axis=0)

        usernum = Dataset.userNum
        auctionnum = Dataset.auctionNum
        bidnum = Dataset.bidNum

        print("(original data) Users/Auction/Bid:", usernum[0], auctionnum[0], bidnum[0])
        print("(final data) Users/Auction/Bid:", usernum[1], auctionnum[1], bidnum[1])
        
        avg_loss = tf.compat.v1.placeholder(tf.float32, [], 'loss')
        tf.compat.v1.summary.scalar('loss', avg_loss)

        validation_HR = tf.compat.v1.placeholder(tf.float32, [], 'validation_HR')
        tf.compat.v1.summary.scalar('validation_HR', validation_HR)
        
        validation_MRR = tf.compat.v1.placeholder(tf.float32, [], 'validation_MRR')
        tf.compat.v1.summary.scalar('validation_MRR', validation_MRR)
        
        validation_NDCG = tf.compat.v1.placeholder(tf.float32, [], 'validation_NDCG')
        tf.compat.v1.summary.scalar('validation_NDCG', validation_NDCG)
        
        
        session.run(tf.compat.v1.local_variables_initializer())
        session.run(tf.compat.v1.global_variables_initializer())
        
        flag = True
        while flag:
          option = input('Query for auction(1) or bidder(2):')
          if option == '1':
            aid = input('Input the id of auction: ')
          else:
            uid = input('Input the id of bidder: ')
            if isdigit(uid):
              uid = int(uid)
              print('The bid history of user',uid,'is ')
              for i in range(len(Dataset.testUserData[int(uid)])):
                  pid = Dataset.product_2_id[Dataset.testUserData[uid][i]['asin']]
                  print(Dataset.testmetaData[pid])



def testRecording(itememb, observed_auctions):
    # if itememb in observed_auctions[0]:
    #     return 1,0
    # else:
        return 0,1


def formerChoose():
    print('------ recommend previous item')

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
    

    parser.add_argument('--user-emb', type=str, default="Complete",
                        help='Select type of user embedding: Complete, Short_term and Long_term')
    parser.add_argument('--loss-f', type=str, default="MetricLearning",
                        help='Select type of loss function: MetricLearning and Inner_product')
    parser.add_argument('--window-size', type=int, default=5,
                        help='ProNADE Input Data Window size(0:Use all product bought before)')
    parser.add_argument('--embed-size', type=int, default=100,
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
    parser.add_argument('--num-units', type=int, default=100,
                        help='the number of hidden unit in lstm model')                 
    
    # Transformer-Parameter
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    return parser.parse_args()

    
if __name__ == "__main__":
    main(parse_args())