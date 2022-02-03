import pickle
import numpy as np
import pandas as pd
import torch
from utils.news_rep import *
import warnings
warnings.filterwarnings('ignore')
from utils.w2v_rep import *
from tqdm import tqdm


file1 = open(".../MIND_dataset/results.txt","w")
pickling_on = open(".../MIND_dataset/Bert4Rec_model_finetuned.pickle","rb")
Model_bert1 = pickle.load(pickling_on)
pickling_on.close()

def zero_pad_post(matrix,max_len):
  pad_length = max_len-len(matrix)
  if pad_length==50:
    return np.zeros((50,300),dtype=float)
  zero_pad = np.zeros((pad_length,300),dtype=float)
  return np.concatenate((matrix,zero_pad))

def zero_pad_pre(matrix,max_len):
  pad_length = max_len-len(matrix)
  zero_pad = np.zeros((pad_length,50,300),dtype=float)
  return np.concatenate((zero_pad,matrix))

dfx = pd.read_csv("/media/sda2/Share/Surupendu/MIND_dataset/MINDlarge_test/behaviors.tsv",sep='\t',header=None)

input = dfx[[0,1,3,4]]
input[3] = input[3].str.split()
input[4] = input[4].str.split()
s5 = []
s6 = []
for i in list(input[4]):
  nes_id = []
  # nes_lab = []
  for j in i:
    nes_id.append(j)
    # nes_lab.append(j.split('-')[1])
  s5.append(nes_id)
  # s6.append(nes_lab)
input[5] = pd.Series(s5)
# input[6] = pd.Series(s6)
input = input.dropna()
input.drop(4,axis = 1, inplace = True)
input.reset_index(inplace = True)
new_3 = []
for lis in list(input[3]):
  new_3.append(lis[:50])
input[3] = pd.Series(new_3)
title_embeddings = embeddings(task='test').find_embeddings()
title_embeddings_train = embeddings().find_embeddings()
title_embeddings_dev = embeddings(task='val').find_embeddings()
title_embeddings.update(title_embeddings_train)
title_embeddings.update(title_embeddings_dev)



class MINDDataset_test():

    def __init__(self,w2v_vectors,test_dataset,transform=None,max_len=200):
        self.w2v_vectors = w2v_vectors
        self.test_dataset = test_dataset
        self.max_len = max_len

    def __getitem__(self,index):
        data = list(self.test_dataset.iloc[index])
        #print(history.shape)
        index_ = data[1]
        history = [zero_pad_post(self.w2v_vectors[p],50) for p in data[3]]
        history = zero_pad_pre(history,50)
        hist_len = len(history)
        mask = torch.LongTensor([0]*(self.max_len-hist_len)+[1 for _ in range(hist_len)])
        candidate_list1 = np.array([zero_pad_post(self.w2v_vectors[k],50) for k in data[4]])
        return index_,mask,history,candidate_list1

    def __len__(self):
        return len(self.test_dataset)

mindDataset = MINDDataset_test(w2v_vectors = title_embeddings,test_dataset=input,max_len=50)
test_loader = torch.utils.data.DataLoader(mindDataset, batch_size=1,shuffle=False)
# Model_bert1 = BERTModel(args)
Model_bert1.eval()
for index,mask,hist,candidates in tqdm(test_loader):
    mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
    user_rep,candidates_rep = Model_bert1(hist.cuda(),mask.cuda(),candidates.cuda())
    scores = torch.bmm(candidates_rep,user_rep.unsqueeze(2))
    pred_prob = torch.sigmoid(scores).view(-1)
    pred_prob = torch.argsort(pred_prob,descending=True)
    ranks = np.zeros(len(pred_prob))
    count = 1
    for i in pred_prob:
      ranks[i] = int(count)
      count+=1
    pred_prob = np.array(ranks).astype(int)
    file1.write(str(index.item())+" "+str(list(pred_prob))+"\n")
file1.close()