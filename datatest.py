from Data_loader.Data import DataSet
import Data_loader.Data_Generator_auction as data
import Model.SharedEmbedding as SE
import Model.test_model as test_model
import os,json,argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import evaluate as me
import matplotlib.pyplot as plt




def train(model, Embed, Dataset, params):
    print('----------------------train function-----------------------------')


    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session:
        # train dataset
        train_dataset = np.array(Dataset.train_data)
        np.random.shuffle(train_dataset)
        
        # eval dataset
        eval_dataset = np.array(Dataset.eval_data)
        np.random.shuffle(eval_dataset)
        
        # shape (2442, 5) 1 3 為list
        train_dataset = np.concatenate((train_dataset,eval_dataset),axis=0)

        timeReport(Dataset.timeList, Dataset.timestampList)

        session.run(tf.compat.v1.local_variables_initializer())
        session.run(tf.compat.v1.global_variables_initializer())
        
        
        # define loss
        losses = []
        pos_losses = []
        neg_losses = []

        # define performance indicator
        HR_list = []
        MRR_list = []
        NDCG_list = []

        total_batch = int(len(train_dataset) / params.batch_size) + 1
        step = 0
        min_loss = 10000.
        best_val = 10000. 
        best_HR = 0.
        best_MRR = 0.
        best_NDCG = 0.
        print('total_batch', total_batch, 'params.epoch:', params.epoch)

        # for e in range(params.epoch):
        #     train_startpos = 0
        #     for b in range(total_batch):
        #         # ((uid, cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_flag, cur_before_pids_attr,current_pid_pos,cur_pids_attr))
        #         u_train,bpp_train,sl_train,bpf_train,cpp_train,cni_train,cbpd_train,cbpo_train,bpd_train,bpo_train \
        #             = data.Get_next_batch(Dataset, train_dataset, train_startpos, params.batch_size)
        #         print(u_train.shape)# 都是64
        #         # _, train_loss, train_pos_loss, train_neg_loss = model.step(session, u_train,bpp_train,sl_train,bpf_train,cpp_train,cni_train,cbpd_train,cbpo_train,bpd_train,bpo_train )
                
def timeReport(times,timestamps):

    plt.hist(times, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Bid Numbers")
    plt.title("Time Distribution historgram")
    plt.show()


    plt.hist(timestamps, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Timestamp")
    plt.ylabel("Bid Numbers")
    plt.title("Timestamp Distribution historgram")
    plt.show()

def main(args):
    if not os.path.isdir(args.model):
        os.mkdir(args.model)

    with open(os.path.join(args.model, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    Dataset = data.Generate_Data(args) # 根據時間分類，在Data中，修改DataSet Class
    Embed = SE.Embedding(Dataset, args)
    model = test_model.Seq(Embed, args)
    train(model, Embed, Dataset, args)


def parse_args():
       
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./Model/psammodel/",
                        help='path to model output directory')
    parser.add_argument('--pig', type=str, default="./picture/",
                        help='path to picture output directory')
    parser.add_argument('--ReviewDataPath', type=str, default="./Dataset/Review/",
    					help='path to the input review dataset')
    parser.add_argument('--MetaDataPath', type=str, default="./Dataset/Meta/",
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
    parser.add_argument('--epoch', type=int, default=100,
                        help='train data epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--validation-batch-size', type=int, default=64,
                        help='the batch size of validation')              
    parser.add_argument('--neg-sample-num', type=int, default=5,
                        help='the number of negative sample')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--depth', type=int, default=20,
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