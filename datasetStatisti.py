import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('dataredPreproMINI1\\train1.txt', header = None,delimiter = "\t", lineterminator='\n')
splitted_data="C:\\Users\\stefano\\OneDrive\\laurea magistrale inf man\\3.1 tirocinio curriculare\\code2vec_path_extractor\\gcjpydataredSPLIT"

def expand2columns(df, col, sep):
    r = df[col].str.split(sep,1)
    
    dfinale=pd.DataFrame.from_records(r)
    return dfinale

#'dataprologin/dataset11.txt'


def file_len(fname):
    with open(fname, encoding="utf8") as f:
        for i, l in enumerate(f):
            #print(i)
            a=i   
            
            if l[0]=="#":
                a=a-1
                pass
            else: 
                pass

            l= a + 1
            
    return l


def avg_line(origin):
    a=0
   
    for root, dirs, files in os.walk(origin, topdown = False): 
        i=0       
        for name in files:
            i=i+1

            doc= os.path.join(root, name)
            a=file_len(doc)+ a
        #avg_len= a/i
        print("total number of line", a)
        #print("avg loc per file:", avg_len)


avg_line(splitted_data) #NUMERO MEDIO DI LINEE PER FILE ESCLUDENDO LE LINEE COMMENTI

data = expand2columns(data, 0, sep=' ')

#df = pd.read_csv('dataprologin\dataset11.txt', sep=" ", error_bad_lines=False)
print("shape dataset",data.shape)
print("number of author",len(data.iloc[:,0].unique()))
print(data.groupby(data.iloc[:,0]).count())

filexauthor=data.groupby(data.iloc[:,0]).count()
filexauthor1=filexauthor.iloc[:,0]
plt.pie(filexauthor1)
plt.legend(data.iloc[:,0].unique())
#plt.show()
#print(data.iloc[:,1])




#print(len(pickle.load( open( "datagrezzo\path2count1.pkl", "rb" ) )))
#print(len(pickle.load( open( "datagrezzo\word2count1.pkl", "rb" ) )))
