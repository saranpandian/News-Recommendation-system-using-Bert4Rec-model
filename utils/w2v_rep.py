import gensim
import numpy as np
import pandas as pd
import re
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
model_path = ".../MIND_dataset/MINDlarge_train/word2vec/word2vec.model"
model_w2v = KeyedVectors.load(model_path, mmap="r")

class extract():
  def removePunctAndNumbers(self,xmlfile):
    res = re.sub(r'[^\w\s]', ' ', str(xmlfile) )
    res = re.sub("\d+", " ", res)
    return res
  def stopword(self,lower_text):
    ext = word_tokenize(lower_text)
    stopword = stopwords.words('english')
    extract=[i for i in ext if i not in stopword]
    return extract
  def extract_text(self,x):
    x=str(x).replace('\n',' ')
    x=self.removePunctAndNumbers(x).lower().strip()
    x=self.stopword(x)
    return x
f = extract()

class embeddings():
  def __init__(self,task = 'train'):
    if task == 'train':
      self.df1 = pd.read_csv("/media/sda2/Share/Surupendu/MIND_dataset/MINDlarge_train/news.tsv",sep='\t',header=None)
    elif task=='test':
      self.df1 = pd.read_csv("/media/sda2/Share/Surupendu/MIND_dataset/MINDlarge_test/news.tsv",sep='\t',header=None)
    elif task=='val':
      self.df1 = pd.read_csv("/media/sda2/Share/Surupendu/MIND_dataset/MINDlarge_dev/news.tsv",sep='\t',header=None)
    self.news_titles = pd.Series(self.df1[3].values,index=self.df1[0]).to_dict()


  def find_embeddings(self):
    preprocessed_titles={}
    for source in list(self.news_titles.keys()):
      preprocessed_titles[source]=f.extract_text(self.news_titles[source])
      if len(preprocessed_titles[source])>50:
        preprocessed_titles[source] = preprocessed_titles[source][:50]
    title_embeddings = {}
    for source in list(preprocessed_titles.keys()):
      temp = []
      for word in preprocessed_titles[source]:
        try:
          temp.append(model_w2v[word])
        except:
          pass
      title_embeddings[source] = np.array(temp)
    t = np.random.randn(50,300)
    title_embeddings['[MASK]'] = t
    return title_embeddings
