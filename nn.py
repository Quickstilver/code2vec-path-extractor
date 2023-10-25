#import functionality

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


import os
import random 
import pickle
import numpy as np
import models
from ray import tune


# setup parameters
config = {
'EMBEDDING_DIM': tune.grid_search([16,32,64]), #8,16,32,64,128 #era 32
'BATCH_SIZE': tune.grid_search([8,16,32,64]), #8,16,32,64,128 #era 8
'N_EPOCHS' : tune.grid_search([300]), #200,300 #era 300
'DROPOUT': tune.grid_search([0.7]) #era 0.7
}


SEED = 1234
CHUNKS = 5 #5 #10
MAX_LENGTH = 5000 ##200 dim originale della rappresentazione
LOG_EVERY = 100 #print log of results after every LOG_EVERY batches
DATASET = 'java14m'
LOG_DIR = 'logs' #/public.hpc/stefanovenv/logs
SAVE_DIR = 'checkpoints'
LOG_PATH = os.path.join(LOG_DIR, f'{DATASET}-log.txt')
MODEL_SAVE_PATH = "/home/students/stefano.balla/stefanovenv/bestMODEL/checkpoints/java14m-model3.pt"     #INSERIRE PATH DEL MODELC HE SI VUOLE TESTARE
MODEL_SAVE_PATH1= "/home/students/stefano.balla/stefanovenv/logs/train_sens_2021-09-06_11-29-00/train_sens_a1bb0_00040_40_BATCH_SIZE=8,DROPOUT=0.7,EMBEDDING_DIM=32,N_EPOCHS=300_2021-09-08_03-09-15/checkpoints/java14m-model.pt"
MODEL_SAVE_PATH2= "/home/students/stefano.balla/stefanovenv/logs/train_sens_2021-09-06_11-29-00/train_sens_a1bb0_00056_56_BATCH_SIZE=16,DROPOUT=0.7,EMBEDDING_DIM=64,N_EPOCHS=300_2021-09-08_19-23-05/checkpoints/java14m-model.pt"

# set random seeds for reproducability

'''
#DATASET PROLOGIN
word2count="dataprologin/word2count1.pkl"
path2count="dataprologin/path2count1.pkl"
target2count="dataprologin/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataprologin/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataprologin/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataprologin/test1.txt"
dimDataset= 17528


#DATASET gcpy
word2count= "data/word2count.pkl"
path2count="data/path2count.pkl"
target2count="data/target2count.pkl"
trainset="/home/students/stefano.balla/stefanovenv/data/train.txt"
valset="/home/students/stefano.balla/stefanovenv/data/val.txt"
testset="/home/students/stefano.balla/stefanovenv/data/test.txt"
dimDataset=280


#DATASET gcpy bilanciato fra le classi in train-test e randomizzato
word2count= "datarandom/word2count1.pkl"
path2count="datarandom/path2count1.pkl"
target2count="datarandom/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datarandom/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datarandom/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datarandom/test1.txt"
dimDataset=274


#DATASET gcpy FORMATTED WITH BLACK
word2count="dataformatted/dataformatted/word2count1.pkl"
path2count="dataformatted/dataformatted/path2count1.pkl"
target2count="dataformatted/dataformatted/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/test1.txt"
dimDataset=280

#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo
word2count="data/word2count1.pkl"
path2count="data/path2count1.pkl"
target2count="data/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/data/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/data/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/data/test1.txt"
dimDataset=778

#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo E FORMATTATO CON BLACK
word2count="datagrezzo/word2count1.pkl"
path2count="datagrezzo/path2count1.pkl"
target2count="datagrezzo/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzo/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzo/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzo/test1.txt"
dimDataset=778


#DATASET gcpy AST non filtrato sui funcdef e classdef gcpygrezzo
word2count="dataAST/word2count1.pkl"
path2count="dataAST/path2count1.pkl"
target2count="dataAST/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataAST/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataAST/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataAST/test1.txt"
dimDataset=666

#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo lunghezza esempi/righe MAX=400 CAMBIARE PAD
word2count="datagrezzolenght400/word2count1.pkl"
path2count="datagrezzolenght400/path2count1.pkl"
target2count="datagrezzolenght400/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzolenght400/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzolenght400/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzolenght400/test1.txt"
dimDataset=778


#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo con lunghezza dei path raddoppiata
word2count="datagrezzopath2/word2count1.pkl"
path2count="datagrezzopath2/path2count1.pkl"
target2count="datagrezzopath2/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzopath2/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzopath2/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzopath2/test1.txt"
dimDataset=778

#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo con FUNZIONE HASH CAMBIATA
word2count="datagrezzoHASH/word2count1.pkl"
path2count="datagrezzoHASH/path2count1.pkl"
target2count="datagrezzoHASH/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzoHASH/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzoHASH/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzoHASH/test1.txt"
dimDataset=778


#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo OVERSAMPLING
word2count="datagrezzoPrepro/word2count1.pkl"
path2count="datagrezzoPrepro/path2count1.pkl"
target2count="datagrezzoPrepro/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzoPrepro/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzoPrepro/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzoPrepro/test1.txt"
dimDataset=3990


#DATASET gcpy non filtrato sui funcdef e classdef gcpygrezzo OVERSAMPLING E FORMATTATO CON BLACK
word2count="datagrezzoFormattedPrepro/word2count1.pkl"
path2count="datagrezzoFormattedPrepro/path2count1.pkl"
target2count="datagrezzoFormattedPrepro/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzoFormattedPrepro/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzoFormattedPrepro/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzoFormattedPrepro/test1.txt"
dimDataset=3990

#DATASET MINI
word2count="datagrezzoMINIPrepro/word2count1.pkl"
path2count="datagrezzoMINIPrepro/path2count1.pkl"
target2count="datagrezzoMINIPrepro/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/datagrezzoMINIPrepro/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/datagrezzoMINIPrepro/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/datagrezzoMINIPrepro/test1.txt"
dimDataset=2144


#DATASET ridotto originale
word2count="dataredPrepro1/word2count1.pkl"
path2count="dataredPrepro1/path2count1.pkl"
target2count="dataredPrepro1/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataredPrepro1/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataredPrepro1/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataredPrepro1/test1.txt"
#dimDataset=1593
dimDataset=354



#DATASET ridotto BLACKED
word2count="dataredPreproBLACKED1/word2count1.pkl"
path2count="dataredPreproBLACKED1/path2count1.pkl"
target2count="dataredPreproBLACKED1/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataredPreproBLACKED1/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataredPreproBLACKED1/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataredPreproBLACKED1/test1.txt"
#dimDataset=1593
dimDataset=354



#DATASET ridotto MINIFIED
word2count="dataredPreproMINI1/word2count1.pkl"
path2count="dataredPreproMINI1/path2count1.pkl"
target2count="dataredPreproMINI1/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/dataredPreproMINI1/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/dataredPreproMINI1/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/dataredPreproMINI1/test1.txt"
#dimDataset=1888
dimDataset=354
'''

#DATASET ridotto blcked
word2count="gcjpyredMINIprocessed/word2count1.pkl"
path2count="gcjpyredMINIprocessed/path2count1.pkl"
target2count="gcjpyredMINIprocessed/target2count1.pkl"
trainset="/home/students/stefano.balla/stefanovenv/gcjpyredMINIprocessed/train1.txt"
valset="/home/students/stefano.balla/stefanovenv/gcjpyredMINIprocessed/val1.txt"
testset="/home/students/stefano.balla/stefanovenv/gcjpyredMINIprocessed/test1.txt"
dimDataset=354


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#load counts of each token in dataset
word2count = pickle.load(open(word2count, "rb" )) #dataformatted/dataformatted/word2count1.pkl
path2count = pickle.load(open( path2count, "rb" )) #dataformatted/dataformatted/path2count1.pkl
target2count = pickle.load(open( target2count, "rb" )) #dataformatted/dataformatted/target2count1.pkl
n_training_examples = dimDataset  #modificare col numero di righeeeeee del training!!!! prologin: 17528, datasetformatted:280


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
    idx2target[v] = k

def first_to_call(config): ##da rimuovere la function e reidentare come se fosse main

    model = models.Code2Vec(len(word2idx), len(path2idx), config['EMBEDDING_DIM'], len(target2idx), config['DROPOUT']) #rimettere tutto senza fig

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda') #'cpu' quando è su laptop, 'cuda' quando su macchina remota

    model = model.to(device)
    criterion = criterion.to(device)

    return model,optimizer,criterion,device #da cancellare

###############START OF DEF FUNCTION####################
def calculate_accuracy(fx, y):
    """
    Calculate top-1 accuracy

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    correct = pred_idxs.eq(y.view_as(pred_idxs)).sum()
    acc = correct.float()/pred_idxs.shape[0]
    return acc

def calculate_f1(fx, y):
    """
    Calculate precision, recall and F1 score
    - Takes top-1 predictions
    - Converts to strings
    - Splits into sub-tokens
    - Calculates TP, FP and FN
    - Calculates precision, recall and F1 score

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    pred_names = [idx2target[i.item()] for i in pred_idxs]
    original_names = [idx2target[i.item()] for i in y]
    true_positive, false_positive, false_negative = 0, 0, 0
    predicted=[]
    original=[]
    len1=0

    
    for p, o in zip(pred_names, original_names):
        
        predicted+=p
        original+=o
        predicted_subtokens = p.split('|')
        
        original_subtokens = o.split('|')
        
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        
        #print(len(predicted))
        
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1, predicted, original

def parse_line(line):
    """
    Takes a string 'x y1,p1,z1 y2,p2,z2 ... yn,pn,zn and splits into name (x) and tree [[y1,p1,z1], ...]
    """
    name, *tree = line.split(' ')
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
     #cast name from int to string
    return name, tree

def file_iterator(file_path):
    """
    Takes a file path and creates and iterator
    For each line in the file, parse into a name and tree
    Pad tree to maximum length
    Yields example:
    - example_name = 'target'
    - example_body = [['left_node','path','right_node'], ...]
    """
    with open(file_path, 'r') as f:

        for line in f:

            #each line is an example

            #each example is made of the function name and then a sequence of triplets
            #the triplets are (node, path, node)

            example_name, example_body = parse_line(line)

            #max length set while preprocessing, make sure none longer

            example_length = len(example_body)

            assert example_length <= MAX_LENGTH
            
            #need to pad all to maximum length

            example_body += [['<pad>', '<pad>', '<pad>']]*(MAX_LENGTH - example_length)
            
            assert len(example_body) == MAX_LENGTH

            yield example_name, example_body, example_length

def numericalize(examples, n):
    """
    Examples are a list of list of lists, i.e. examples[0] = [['left_node','path','right_node'], ...]
    n is how many batches we are getting our of `examples`
    
    Get a batch of raw (still strings) examples
    Create tensors to store them all
    Numericalize each raw example within the batch and convert whole batch tensor
    Yield tensor batch
    """
    p=config['BATCH_SIZE'] 
    pp= p['grid_search'][0]
    assert n*pp<= len(examples) #rimuovere config

    for i in range(n):
        

        #get the raw data
        ## sostituire pp con config['BATCH_SIZE']
        raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[pp*i:pp*(i+1)]) #rimuovere config
        
        #create a tensor to store the batch
        
        tensor_n = torch.zeros(pp).long() #name sostituire pp con config['BATCH_SIZE']
        tensor_l = torch.zeros((pp, MAX_LENGTH)).long() #left node
        tensor_p = torch.zeros((pp, MAX_LENGTH)).long() #path
        tensor_r = torch.zeros((pp, MAX_LENGTH)).long() #right node
        mask = torch.ones((pp, MAX_LENGTH)).float() #mask
        
        #for each example in our raw data
        
        for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
            
            #convert to idxs using vocab
            #use <unk> tokens if item doesn't exist inside vocab
            temp_n = target2idx.get(name, target2idx['<unk>'])
            temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
            
            #store idxs inside tensors
            tensor_n[j] = temp_n
            tensor_l[j,:] = torch.LongTensor(temp_l)
            tensor_p[j,:] = torch.LongTensor(temp_p)
            tensor_r[j,:] = torch.LongTensor(temp_r)   #LongTensor
            
            #create masks
            mask[j, length:] = 0

        yield tensor_n, tensor_l, tensor_p, tensor_r, mask

def get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, optimizer=None):
    """
    Takes inputs, calculates loss, accuracy and other metrics, then calculates gradients and updates parameters

    if optimizer is None, then we are doing evaluation so no gradients are calculated and no parameters are updated
    """

    if optimizer is not None:
        optimizer.zero_grad()

    fx = model(tensor_l, tensor_p, tensor_r)

    loss = criterion(fx, tensor_n)

    acc = calculate_accuracy(fx, tensor_n)
    precision, recall, f1, predicted,original = calculate_f1(fx, tensor_n)
    
    if optimizer is not None:
        loss.backward()
        optimizer.step()   
    

    return loss.item(), acc.item(), precision, recall, f1, predicted,original

def train(device,model, file_path, optimizer, criterion): #remove device
    """
    Training loop for the model
    Dataset is too large to fit in memory, so we stream it
    Get BATCH_SIZE * CHUNKS examples at a time (default = 1024 * 10 = 10,240)
    Shuffle the BATCH_SIZE * CHUNKS examples
    Convert raw string examples into numericalized tensors
    Get metrics and update model parameters

    Once we near end of file, may have less than BATCH_SIZE * CHUNKS examples left, but still want to use
    So we calculate number of remaining whole batches (len(examples)//BATCH_SIZE) then do that many updates
    """
    
    n_batches = 0
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0
    
    model.train()
    
    examples = []
    
    for example_name, example_body, example_length in file_iterator(file_path):

        examples.append((example_name, example_body, example_length))
        ll=config['BATCH_SIZE'] #rimettere batch size al posto di ll
        #print( 'ooooooooooooooooooooooooooooooooo', ll['grid_search'])
        lll=ll['grid_search'][0] ##ottiene la batch size in formato int

    
        if len(examples) >= (lll * CHUNKS):

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS):

                #place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                #put into model
                loss, acc, p, r, f1,predicted,original = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, optimizer)

                epoch_loss += loss
                epoch_acc += acc
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1
                
                n_batches += 1
                                    
                if n_batches % LOG_EVERY == 0:
            
                    loss = epoch_loss / n_batches
                    acc = epoch_acc / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches
                    l=config['BATCH_SIZE'] #sostituire l nella riga di log = f... con BATCH_SIZE
                    l=l['grid_search'][0]
                    

                    log = f'\t| Batches: {n_batches} | Completion: {((n_batches*l)/n_training_examples)*100:03.3f}% |\n'
                    log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log+'\n')
                    print(log)

            examples = []
                            
        else:
            pass
      
    #outside of `file_iterator`, but will probably still have some examples left over
    random.shuffle(examples)

    #get amount of batches we have left
    n = len(examples)//lll #config['BATCH_SIZE']

    #train with remaining batches
    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n):
            
        #place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)
            
        #put into model
                
        loss, acc, p, r, f1,predicted,original = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, optimizer)

        epoch_loss += loss
        epoch_acc += acc
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1
        
        n_batches = n_batches + 1
        
        print("batch",n_batches)


    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

def evaluate(device,model, file_path, criterion): #remove device
    """
    Similar to training loop, but we do not pass optimizer to get_metrics
    Also wrap get_metrics in `torch.no_grad` to avoid calculating gradients
    """

    n_batches = 0
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0
    predizione=[]
    orig=[]
    
    model.eval()
    
    examples = []
    
    
    for example_name, example_body, example_length in file_iterator(file_path):
        
        examples.append((example_name, example_body, example_length))
        p=config['BATCH_SIZE'] 
        pp= p['grid_search'][0]
        

        if len(examples) >= (pp* CHUNKS): #rimettere batch size BATCH_SIZE
            

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS):

                #place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                #put into model
                with torch.no_grad():
                    loss, acc, p, r, f1,predicted,original = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion)
                

                epoch_loss += loss
                epoch_acc += acc
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1

                predizione+=predicted
                orig+=original



                
                n_batches += 1
                 
                if n_batches % LOG_EVERY == 0:
            
                    loss = epoch_loss / n_batches
                    acc = epoch_acc / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches


                    log = f'\t| Batches: {n_batches} |\n'
                    log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log+'\n')
                    print(log)

            examples = []
                            
        else:
            pass
      
    #outside of for line in f, but will still have some examples left over

    random.shuffle(examples)


    n = len(examples)//pp #config['BATCH_SIZE']
    print('division',n)
    print('number examples',len(examples))

    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n):
            
        #place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)
            
        #put into model
        with torch.no_grad():
            loss, acc, p, r, f1,predicted,original = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion)

        epoch_loss += loss
        epoch_acc += acc
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1
        
        n_batches += 1
        predizione+=predicted
        orig+=original


        
        
  
        
    #print(metrics.confusion_matrix(orig,predizione))
    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

###################END OF DEF FUNCTION, HERE THE EXECUTION START#######




################# TUNING##############

def execution():
    #tuning parameters execution
    analysis = tune.run(train_sens,resources_per_trial={'gpu': 1}, config=config, local_dir="./stefanovenv/logs"  ) #'/home/students/stefano.balla/stefanovenv/logs'

    #best_trial = analysis.get_best_trial('mean_accuracy')
    #print("Best trial config: {}".format(best_trial.config))

    df_result = analysis.results_df

    max= df_result["mean_accuracy"].max()
    print('Best configuration ==> ')
    print(df_result[df_result["mean_accuracy"]==max])
 
###################END TUNING#############

##DEFINE A NEW FUNCTION trainsens for the tuning process
def train_sens(config):

    #rimuovere model da qua e lasciarla solo in first call
    #model = models.Code2Vec(len(word2idx), len(path2idx), config['EMBEDDING_DIM'], len(target2idx), config['DROPOUT'])
    model,optimizer,criterion,device= first_to_call(config)

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    if not os.path.isdir(f'{LOG_DIR}'):
        os.makedirs(f'{LOG_DIR}')

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    
    for i in range(5): ##iteration erano 5
        


        for epoch in range(config['N_EPOCHS']):

            log = f'Epoch: {epoch+1:02} - Training'
            with open(LOG_PATH, 'a+') as f:
                f.write(log+'\n')
            print(log)
            
            #remove device dalla chiamata di train
            train_loss, train_acc, train_p, train_r, train_f1 = train(device,model,trainset, optimizer, criterion) #'code2vec-master/data/java14m/train1.txt' /home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/train1.txt'
            
            log = f'Epoch: {epoch+1:02} - Validation'
            with open(LOG_PATH, 'a+') as f:
                f.write(log+'\n')
            print(log)
            
            #remove device dalla chiamata di evaluate
            valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(device,model,valset, criterion) 
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                '''
                for f in LOG_DIR:
                    try:
                        shutil.rmtree(f)
                    except OSError as e:
                        print("Error: %s : %s" % (f, e.strerror))
                '''

                torch.save(model.state_dict(), MODEL_SAVE_PATH) #MODEL_SAVE PATH
            
            log = f'| Epoch: {epoch+1:02} |\n'
            log += f'| Train Loss: {train_loss:.3f} | Train Precision: {train_p:.3f} | Train Recall: {train_r:.3f} | Train F1: {train_f1:.3f} | Train Acc: {train_acc*100:.2f}% |\n'
            log += f'| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p:.3f} | Val. Recall: {valid_r:.3f} | Val. F1: {valid_f1:.3f} | Val. Acc: {valid_acc*100:.2f}% |'
            with open(LOG_PATH, 'a+') as f:
                f.write(log+'\n')
            print(log)


    tune.report(mean_accuracy=valid_acc*100) ####tune aggiunto

    log = 'Testing'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)


    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    #rimuovere device da qui e da train
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(device,model,testset, criterion) #  java14m.val.c2v ##'code2vec-master/data/java14m/test1.txt' '/home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/test1.txt'

    log = f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p:.3f} | Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f} | Test Acc: {test_acc*100:.2f}% |' 
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

    
def testing(): 
    model = models.Code2Vec(len(word2idx), len(path2idx), 32, len(target2idx), 0.7) #METTERE EMBEDDING E DROPOUT DEL BEST EXPERIMENT!  
    # load check point
    checkpoint = model.load_state_dict(torch.load(MODEL_SAVE_PATH1))
    criterion = nn.CrossEntropyLoss()
    # return model, optimizer, epoch value, min validation loss  

    device = torch.device('cuda')
    model = model.to(device)
    criterion = criterion.to(device)
    #rimuovere device da qui e da train
    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(device,model,valset, criterion) 
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(device,model,testset, criterion) #  java14m.val.c2v ##'code2vec-master/data/java14m/test1.txt' '/home/students/stefano.balla/stefanovenv/dataformatted/dataformatted/test1.txt'
    
    log = 'Validation \n'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    
    log += f'| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p:.3f} | Val. Recall: {valid_r:.3f} | Val. F1: {valid_f1:.3f} | Val. Acc: {valid_acc*100:.2f}% | \n'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

    log = 'Testing \n'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')

    log += f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p:.3f} | Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f} | Test Acc: {test_acc*100:.2f}% | \n' 
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)


execution()
#testing()