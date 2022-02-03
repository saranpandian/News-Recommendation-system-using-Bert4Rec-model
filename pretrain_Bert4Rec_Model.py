import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
from w2v_rep import *
# from bert_model import *
from model_trainer import *
from options import *
import pickle
from utils.Early_stopping import *
import numpy as np
import pandas as pd
import torch

print("creating dataset...")
patience = 4
early_stopping = EarlyStopping(patience=patience, verbose=True)

dfx = pd.read_csv(".../MINDlarge_train/behaviors.tsv",sep='\t',header=None)
# dfx1 = pd.read_csv("/media/sda2/Share/Surupendu/MIND_dataset/MINDlarge_dev/behaviors.tsv",sep='\t',header=None)
input = dfx[[1,3,4]]
input[3] = input[3].str.split()
input.dropna(inplace=True)

empty = []
for i in list(input[3]):
  empty.extend(i)
# for i in list(input1[3]):
#   empty.extend(i)
news_ids = np.unique(empty)
args.bert_num_items =len(news_ids)

input = input[3]
new_3 = []
for lis in list(input):
  new_3.append(lis[:50])

id_dict = {}
count = 1
for id in news_ids:
  id_dict[id] = count
  count+=1


ce = nn.CrossEntropyLoss()
def calculate_loss(batch):
    seqs, labels = batch
    logits = seqs.view(-1, seqs.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = ce(logits, labels)
    return loss


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

print("creating title embeddings for train")
title_embeddings = embeddings().find_embeddings()

# print("creating title embeddings for val")
# title_embeddings_val = embeddings(task='val').find_embeddings()

class BertDataLoader():
    def __init__(self,title_embeddings,id_dict,mask_token,mask_prob,seed,input,max_len,transform=None):
        self.mask_token = mask_token
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.random_num = random.Random(seed)
        self.title_embeddings = title_embeddings
        self.input = list(input)
        self.id_dict = id_dict
    def __getitem__(self,index):
        history = self.input[index]
        tokens = []
        labels = []
        hist_len = len(history)
        for s in history:
            prob = self.random_num.random()
            if prob < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(self.id_dict[s])
            else:
                tokens.append(s)
                labels.append(0)
        history1 = [zero_pad_post(self.title_embeddings[p],50) for p in tokens]
        history1 = np.array(zero_pad_pre(history1,self.max_len))
        # labels = [self.id_dict[q] for q in labels]

        mask = torch.LongTensor([0]*(self.max_len-hist_len)+[1 for _ in range(hist_len)])
        mask_len = self.max_len - len(labels)
        labels = np.array([0] * mask_len + labels)
        return history1,mask,labels

    def __len__(self):
        return len(self.input)

print("loading data.....")
mindDataset = BertDataLoader(title_embeddings,id_dict,mask_token="[MASK]",mask_prob=0.3,seed=5,input = new_3,max_len=50)

train_loader = torch.utils.data.DataLoader(mindDataset, batch_size=16,shuffle=True)

mindDataset_val = BertDataLoader(title_embeddings_val,id_dict,mask_token="[MASK]",mask_prob=0.3,seed=5,input = new_31,max_len=50)

val_loader = torch.utils.data.DataLoader(mindDataset_val, batch_size=16,shuffle=True)
print("training starts")
Model = model(args).cuda()
accumulation_steps = 8
optimizer = optim.Adam(Model.parameters(),lr = 1e-4)
out = nn.Linear(args.bert_hidden_units, args.bert_num_items + 1).cuda()
for i in range(3):
    loss_hist = []
    Model.train()
    for j,x in enumerate(tqdm(train_loader)):
      mask = x[1].cuda()
      mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
      y = Model(x[0].cuda(),mask)
      y = out(y)
      y = F.softmax(y,dim=2)
      optimizer.zero_grad()
      loss = calculate_loss((y,x[2].cuda()))
      loss = loss / accumulation_steps
      loss.backward()
      if (j+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        Model.zero_grad()
      # torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.0)
      # optimizer.step()
      loss_hist.append(loss.item())

    Model.eval()
    valid_losses = []
    for x in tqdm(val_loader):
      mask = x[1].cuda()
      mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
      y = Model(x[0].cuda(),mask)
      y = out(y)
      y = F.softmax(y,dim=2)
      loss = calculate_loss((y,x[2].cuda()))
      loss = loss / accumulation_steps
      valid_losses.append(loss.item())

    loss_train = np.sum(loss_hist)/len(train_loader)
    loss_val = np.sum(valid_losses)/len(val_loader)
    print("Epoch number {}\n Current loss {}\n".format(i,loss_train))
    pickling_on = open(".../MIND_dataset/pretrained_Masked_Model.pickle","wb")
    pickle.dump(Model,pickling_on)
    pickling_on.close()
    early_stopping(loss_val, Model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
