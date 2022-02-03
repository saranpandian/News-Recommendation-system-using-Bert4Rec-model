from bert_model import *
from options import *
import random

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

class News_Representation(nn.Module):
    def __init__(self):
        super(News_Representation, self).__init__()
        self.conv1d = nn.Conv1d(300,args.bert_hidden_units,3,padding=1,stride=1)
        self.cat_lin = nn.Linear(300,args.bert_hidden_units)
        self.attn = nn.Linear(50,50)
        self.linear_in = nn.Linear(args.bert_hidden_units,1 , bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.attn1 = nn.Linear(100,100)
        # self.linear_in1 = nn.Linear(100,1 , bias=False)

    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)
        # y = y.type(torch.FloatTensor)
        # y = self.cat_lin(y)
        x = x.transpose(2,1)
        l1 = self.conv1d(x)
        l1 = self.relu(l1)
        l2 = self.attn(l1)
        l2 = l2.transpose(2,1)
        l2 = self.tanh(l2)
        l2 = self.linear_in(l2)
        attn_weights = F.softmax(l2,dim=1)
        attn_applied = torch.bmm(l1,attn_weights).squeeze(2)
        return attn_applied


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class model(nn.Module):
    def __init__(self,args):
        super(model, self).__init__()
        fix_random_seed_as(1)
        max_len = args.bert_max_len
        num_items = 1
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        self.NR1=News_Representation()
        self.TD1=TimeDistributed(self.NR1,batch_first=True)
        # self.bert = BERT(args)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self,x,mask):
        nr=self.TD1(x)  #y->[16,5,30,300] -> [16,5,300]
        for transformer in self.transformer_blocks:
            x = transformer.forward(nr, mask)
        return nr

