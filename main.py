import random
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
from utils.options import *
import pickle
from utils.Early_stopping import *
import numpy as np
import torch
from utils.model_trainer import *

print("creating dataset...")
patience = 4
early_stopping = EarlyStopping(patience=patience, verbose=True)

def create_dataset(data):
  temp1 = data[2][:50]
  filtered_0 = list(filter(lambda x:x[1]=='0', zip(data[3],data[4])))
  filtered_1 = list(filter(lambda x:x[1]=='1', zip(data[3],data[4])))
  decomposed = []
  for m in filtered_1:
    temp = filtered_0[:]
    if len(temp)>4:
      temp2 = random.sample(temp, 4)
    else:
      temp2 = random.sample(temp*(4//len(temp)+1),4)

    temp2.append(m)
    random.shuffle(temp2)
    decomposed.append(temp2)
  return temp1,decomposed

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

dfx = pd.read_csv(".../MIND_dataset/MINDlarge_train/behaviors.tsv",sep='\t',header=None)
input = dfx[[1,3,4]]
input[3] = input[3].str.split()
input[4] = input[4].str.split()
s5 = []
s6 = []
for i in list(input[4]):
  nes_id = []
  nes_lab = []
  for j in i:
    nes_id.append(j.split('-')[0])
    nes_lab.append(j.split('-')[1])
  s5.append(nes_id)
  s6.append(nes_lab)
input[5] = pd.Series(s5)
input[6] = pd.Series(s6)
input = input.dropna()
input.drop(4,axis = 1, inplace = True)
input.reset_index(inplace = True)
new_3 = []
for lis in list(input[3]):
  new_3.append(lis[:50])
input[3] = pd.Series(new_3)

train_dataset = []
for user in range(len(input)):
  hist,decomposed = create_dataset(list(input.iloc[user]))
  for samp in decomposed:
    temp3,temp4 = zip(*samp)
    train_dataset.append(tuple((hist,temp3,temp4)))


dfx_val = pd.read_csv(".../MIND_dataset/MINDlarge_dev/behaviors.tsv",sep='\t',header=None)
input_val = dfx_val[[1,3,4]]
input_val[3] = input_val[3].str.split()
input_val[4] = input_val[4].str.split()
s5 = []
s6 = []
for i in list(input_val[4]):
  nes_id = []
  nes_lab = []
  for j in i:
    nes_id.append(j.split('-')[0])
    nes_lab.append(j.split('-')[1])
  s5.append(nes_id)
  s6.append(nes_lab)
input_val[5] = pd.Series(s5)
input_val[6] = pd.Series(s6)
input_val = input_val.dropna()
input_val.drop(4,axis = 1, inplace = True)
input_val.reset_index(inplace = True)
new_3 = []
for lis in list(input_val[3]):
  new_3.append(lis[-50:])
input_val[3] = pd.Series(new_3)

dfx_val = pd.read_csv(".../MIND_dataset/MINDlarge_dev/behaviors.tsv",sep='\t',header=None)
input = dfx_val[[1,3,4]]
input[3] = input[3].str.split()
input[4] = input[4].str.split()
s5 = []
s6 = []
for i in list(input[4]):
  nes_id = []
  nes_lab = []
  for j in i:
    nes_id.append(j.split('-')[0])
    nes_lab.append(j.split('-')[1])
  s5.append(nes_id)
  s6.append(nes_lab)
input[5] = pd.Series(s5)
input[6] = pd.Series(s6)
input = input.dropna()
input.drop(4,axis = 1, inplace = True)
input.reset_index(inplace = True)
new_3 = []
for lis in list(input[3]):
  new_3.append(lis[-200:])
input[3] = pd.Series(new_3)

val_dataset = []
for user in range(len(input)):
  hist,decomposed = create_dataset(list(input.iloc[user]))
  for samp in decomposed:
    temp3,temp4 = zip(*samp)
    val_dataset.append(tuple((hist,temp3,temp4)))

class MINDDataset():
    def __init__(self,w2v_vectors,train_dataset,transform=None,max_len=200):
        self.w2v_vectors = w2v_vectors
        self.train_dataset = train_dataset
        self.max_len = max_len

    def __getitem__(self,index):
        data = self.train_dataset[index]
        history = data[0]
        rec_list = data[1]
        label = data[2]
        hist_len = len(history)
        #print(history.shape)
        history = [zero_pad_post(self.w2v_vectors[p],50) for p in history]
        history = zero_pad_pre(history,50)
        rec_list_embed = np.array([zero_pad_post(self.w2v_vectors[q],50) for q in rec_list])
        label = np.argmax(np.array(label))
        mask = torch.LongTensor([0]*(self.max_len-hist_len)+[1 for _ in range(hist_len)])
        return history,mask,rec_list_embed,label

    def __len__(self):
        return len(self.train_dataset)


class MINDDataset_val():
    def __init__(self,w2v_vectors,test_dataset,transform=None,max_len=200):
        self.w2v_vectors = w2v_vectors
        self.test_dataset = test_dataset
        self.max_len = max_len

    def __getitem__(self,index):
        data = self.test_dataset[index]
        history = data[0]
        rec_list = data[1]
        label = data[2]
        hist_len = len(history)
        #print(history.shape)
        history = [zero_pad_post(self.w2v_vectors[p],50) for p in history]
        history = zero_pad_pre(history,200)
        rec_list_embed = np.array([zero_pad_post(self.w2v_vectors[q],50) for q in rec_list])
        label = np.argmax(np.array(label))
        mask = torch.LongTensor([0]*(self.max_len-hist_len)+[1 for _ in range(hist_len)])
        return history,mask,rec_list_embed,label

    def __len__(self):
        return len(self.test_dataset)

class MINDDataset_val():

    def __init__(self,w2v_vectors,test_dataset,transform=None):
        self.w2v_vectors = w2v_vectors
        self.test_dataset = test_dataset

    def __getitem__(self,index):
        data = list(self.test_dataset.iloc[index])
        #print(history.shape)
        history = [zero_pad_post(self.w2v_vectors[p],50) for p in data[2]]
        history = zero_pad_pre(history,50)

        candidate_list1 = np.array([zero_pad_post(self.w2v_vectors[k],50) for k in data[3]])
        ground_truth = np.array([int(x) for x in data[4]])
        return history,candidate_list1,ground_truth

    def __len__(self):
        return len(self.test_dataset)


print("creating title embeddings for train")
title_embeddings = embeddings().find_embeddings()

print("creating title embeddings for val")
title_embeddings_val = embeddings(task='val').find_embeddings()

mindDataset = MINDDataset(w2v_vectors = title_embeddings,train_dataset=train_dataset,max_len=50)
train_loader = torch.utils.data.DataLoader(mindDataset, batch_size=128,shuffle=True)

mindDataset_val = MINDDataset_val(w2v_vectors = title_embeddings_val,test_dataset=val_dataset,max_len=200)
val_loader = torch.utils.data.DataLoader(mindDataset_val, batch_size=128,shuffle=True)

print("training starts")
MODEL = BERTModel(args).cuda()
# Model=model(input_channels, n_classes, channel_sizes, kernel_size=kernel_size).cuda()
MODEL.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(MODEL.parameters(),lr = 1e-5)
# loss_history=[]
for i in range(4):
    loss_hist = []
    MODEL.train()
    for data in tqdm(train_loader):
      user_his,mask,rec_list,label=Variable(data[0].cuda()),Variable(data[1].cuda()),Variable(data[2].cuda()),Variable(data[3].cuda())
      mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
      UR,NR = MODEL(user_his.cuda(),mask.cuda(),rec_list.cuda())
      s=torch.bmm(NR,UR.unsqueeze(2))
      output=F.softmax(s)
      optimizer.zero_grad()
      loss=criterion(output.squeeze(),label.cuda())
      loss.backward()
      torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
      optimizer.step()
      loss_hist.append(loss.item())

    MODEL.eval()
    valid_losses = []
    for data in tqdm(val_loader):
      # forward pass: compute predicted outputs by passing inputs to the model
      user_his_val,mask,rec_list_val,label = Variable(data[0].cuda()),Variable(data[1].cuda()),Variable(data[2].cuda()),Variable(data[3].cuda())
      mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
      UR,NR = MODEL(user_his.cuda(),mask.cuda(),rec_list.cuda())

      s=torch.bmm(NR,UR.unsqueeze(2))
      output=F.softmax(s)
      loss_va=criterion(output.squeeze(),label_val.cuda())
      valid_losses.append(loss_va.item())

    loss_train = np.sum(loss_hist)/len(train_loader)
    loss_val = np.sum(valid_losses)/len(val_loader)
    early_stopping(loss_val, MODEL)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    print("Epoch number {}\n Current loss {}\n".format(i,loss_train))

    pickling_on = open(".../MIND_dataset/Bert4Rec_model_finetuned.pickle","wb")
    pickle.dump(MODEL,pickling_on)
    pickling_on.close()