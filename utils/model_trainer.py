from news_rep import *
from w2v_rep import *
from Transformer_Blocks import *
from options import *
import torch
import random
import pickle

# BERT_Model = model(args)
# PATH = "/media/sda2/Share/Surupendu/MIND_dataset/bert_model_large.pickle"
# BERT_Model.load_state_dict(torch.load(PATH))

pickling_on = open(".../MIND_dataset/pretrained_Masked_Model.pickle","rb")
model_bert = pickle.load(pickling_on)
# inp = torch.randn((16,5,50,300))
# NR = model_bert.TD1(inp)
# print(NR.shape)
pickling_on.close()
class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel,self).__init__()
        self.NR2=News_Representation()
        self.TD2=TimeDistributed(self.NR2,batch_first=True)
        # self.out = nn.Linear(args.bert_hidden_units, args.bert_num_items + 1)

    def forward(self,x,mask,y):
        with torch.no_grad():
           x = model_bert(x,mask)
        y = self.TD2(y.cuda())
        x = torch.mean(x,dim=1)
        return x,y