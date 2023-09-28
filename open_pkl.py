

import pandas as pd
import pickle


word2count = pickle.load(open("datasets\\processed_dataset\\gcjpyredAST\\word2count1.pkl", "rb" )) #dataformatted/dataformatted/word2count1.pkl
path2count = pickle.load(open( "datasets\\processed_dataset\\gcjpyredAST\\path2count1.pkl", "rb" )) #dataformatted/dataformatted/path2count1.pkl
target2count = pickle.load(open( "datasets\\processed_dataset\\gcjpyredAST\\target2count1.pkl", "rb" )) #dataformatted/dataformatted/target2count1.pkl



# create vocabularies, initialized with unk and pad tokens
word2idx = {'<unk>': 0, '<pad>': 1}
path2idx = {'<unk>': 0, '<pad>': 1 }
target2idx = {'<unk>': 0, '<pad>': 1}

idx2word = {}
idx2path = {}
idx2target = {}

for w in word2count.keys():
    word2idx[w] = len(word2idx)
    
for k, v in word2idx.items():
    idx2word[v] = k
    
for p in path2count.keys():
    path2idx[p] = len(path2idx)
    
for k, v in path2idx.items():
    idx2path[v] = k
    
for t in target2count.keys():
    target2idx[t] = len(target2idx)
   

for k, v in target2idx.items():
    print(k)
    idx2target[v] = k

#print(len(word2idx), len(path2idx), len(target2idx))