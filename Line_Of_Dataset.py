import os
from collections import Counter
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import re

def convert_triplets(triplets):
    converted = []
    for triplet in triplets:
        # regex pattern that captures the three parts of the triplet separately
        pattern = r"(.*?),(b\"[^\"]*\"),(.*)"
        matches = re.match(pattern, triplet)
        if matches:
            # Append a list of the three parts to converted
            converted.append([matches.group(1), matches.group(2), matches.group(3)])
    return converted

def find_triplets(input_string):
    # regex pattern that captures triplets in the form "string, b"[...]", string"
    pattern = r"([^,]*,b\"[^\"]*\",[^,\s]*)"
    matches = re.findall(pattern, input_string)
    matches = [match.lstrip() for match in matches]
    return matches

def split_string(input_string):
    # Split the string at the first occurrence of whitespace
    split_list = input_string.split(' ', 1)

    # If there is no whitespace in the string, return the whole string as the first item and an empty string as the second
    if len(split_list) == 1:
        return split_list[0], ''
    else:
        return split_list[0], split_list[1]

def parse_line(line):
    """
    Takes a string 'x y1,p1,z1 y2,p2,z2 ... yn,pn,zn and splits into name (x) and tree [[y1,p1,z1], ...]
    """
    name, *tree = line.split(' ')
    #print(tree[0])
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
    return name, tree


def create_dict(dest1):
    curr=os.getcwd()
    datar= os.path.join(curr, dest1, "train1.txt")

    with open(datar, 'r') as file_:
        listname=[]
        listpath=[]
        listtoken=[]
        
        for line in file_:
            line = line. rstrip('\n')
            name,tree = split_string(line)
            matches = find_triplets(tree)
            elements = convert_triplets(matches)

            for el in elements:
                listtoken.append(el[0])    #inserisco lo start token e l'end token in una lista per contare le occorrenze
                listtoken.append(el[2])
                listpath.append(el[1])  #inserisco il path in una lista per contare le occorrenze

            listname.append(name)       #inserisco il target in una lsta per contare le occorrenze

        target2count=dict(Counter(listname))
        path2count= dict(Counter(listpath))
        word2count= dict(Counter(listtoken))


    with open(os.path.join(dest1,'target2count1.pkl'),'wb') as file: 
        pickle.dump(target2count, file)

    with open(os.path.join(dest1,'path2count1.pkl'), 'wb') as file: 
        pickle.dump(path2count, file)

    with open(os.path.join(dest1,'word2count1.pkl'), 'wb') as file: 
        pickle.dump(word2count, file)
    
    
    print("dict created")


def expand2columns(df, col, sep):

    r = df[col].str.split(sep,1)
    dfinale=pd.DataFrame.from_records(r)

    return dfinale


def obtain_label(data):

    df= pd.DataFrame(data)
    datanew = expand2columns(df,0,sep=' ')
    label = datanew.iloc[:,0]

    return label


def obtain_X(data):

    df= pd.DataFrame(data)
    datanew = expand2columns(df,0,sep=' ')

    return datanew


def preprocessing(trainset):

    ros = RandomOverSampler()
    y=obtain_label(trainset)
    X= obtain_X(trainset)
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros


def split_train_test_val(pre, dest1):

    curr = os.getcwd()
    datar = os.path.join(curr, dest1, "datasetgrezzo.txt")

    train, test, val = "train1.txt", "test1.txt", "val1.txt"
    train = os.path.join(dest1, train)
    test = os.path.join(dest1, test)
    val = os.path.join(dest1, val)

    with open(datar, 'r') as f: 
        data = np.array(f.read().splitlines())

    y=obtain_label(data)
    data_train, data_val, y_train, y_val = train_test_split(data,y,test_size=0.2,stratify=y)
    y=obtain_label(data_train)
    data_train, data_test, y_train, y_test = train_test_split(data_train,y,test_size=0.25,stratify=y)
    
    print('train',y_train.value_counts(),'val',y_val.value_counts(),'test',y_test.value_counts())
    
    if pre:
        data_train= preprocessing(data_train)  #rimuovere se non si vuole fare l'oversapling 
        with open(train, 'w', encoding="utf-8") as dataset:
            text = '\n'.join(f'{row[0]} {row[1]}' for _, row in data_train.iterrows())
            dataset.write(text + '\n')

    if not pre: 
        print('preprocessing offline')
        with open(train, 'w', encoding="utf-8") as dataset:
            dataset.write('\n'.join(data_train))

    print("n_examples:",len(data_train))

    with open(os.path.join(dest1,'train1.pkl'), 'wb') as f:  
        pickle.dump(data_train, f)
        
    with open(test, 'w', encoding="utf-8") as dataset1:
        dataset1.write('\n'.join(data_test))

    with open(os.path.join(dest1,'test1.pkl'), 'wb') as f1:
        pickle.dump(data_test, f1)
    
    with open(val, 'w', encoding="utf-8") as dataset2:
        dataset2.write('\n'.join(data_val))

    with open(os.path.join(dest1,'val1.pkl'), 'wb') as f2:
        pickle.dump(data_val, f2)        
    
    print("dataset splitted in train test val")
    return data_train, data_test, data_val
    
#split_train_test_val(True,'cancella')
#create_dict('cancella')

        