#!/usr/bin/env python
# coding: utf-8
'''fun: sentiment analysis'''


import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
# import pandas as pd
from DataLoader import MovieDataset
from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description = "LSTM Sentiment Analysis")
parser.add_argument("-embedding_dim", dest="embedding_dim", type=int, default=100, help="dim of embedding")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=50, help="number of epoches")
parser.add_argument("-load_cpt", dest="load_cpt", type=int, default=False, help="load checkpoint")
parser.add_argument("-hidden_dim", dest="hidden_dim", type=int, default=50, help="hidden_dim")
args = parser.parse_args()

'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)


def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate
    if epoch > 5:
        lr = learning_rate / 10
    elif epoch > 10:
        lr = learning_rate / 100
    elif epoch > 20:
        lr = learning_rate / 1000
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    ## alternative method
    # torch.cuda.set_device(device=0) ## choose gpu number
    print('device: ', device)

    ## ---------------------------------------------------------
    ## please change the parameter settings by yourselves
    ## ---------------------------------------------------------
    mode = 'train'
    Batch_size =300
    n_layers = 1 ## choose 1-3 layers

    ## input seq length aligned with data pre-processing
    input_len = 150

    ## word embedding length
    embedding_dim = args.embedding_dim #50

    # lstm hidden dim
    hidden_dim = args.hidden_dim #50
    # binary cross entropy
    output_size = 1
    num_epoches = args.num_epoches
    ## please change the learning rate by youself
    learning_rate = 0.002
    # gradient clipping
    clip = 5
    load_cpt = args.load_cpt #True
    ckp_path = 'cpt/name.pt'
    # embedding_matrix = None
    ## use pre-train Glove embedding or not?
    pretrain = False

    ##-----------------------------------------------------------------------
    ## Bonus (5%): complete code to add GloVe embedding file path below.
    ## Download Glove embedding from https://nlp.stanford.edu/data/glove.6B.zip
    ## "embedding_dim" defined above shoud be aligned with the dimension of GloVe embedddings
    ## if you do not want bonus, you can skip it.
    ##-----------------------------------------------------------------------
    glove_file = 'path/glove.6B.200d.txt' ## change by yourself
    

    ## ---------------------------------------------------------
    ## step 1: create data loader in DataLoader.py
    ## complete code in DataLoader.py (not Below)
    ## ---------------------------------------------------------
    
    
    ## step 2: load training and test data from data loader [it is Done]
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size,\
                                    shuffle=True,num_workers=1)
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size,\
                                shuffle=False,num_workers=1)


    ## step 3: [Bonus] read tokens and load pre-train embedding [it is Done]
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    ## -----------------------------------------------
    ## step 4: import model from LSTM.py
    ## complete the code in "def forward(self, x)" in LSTM.py file
    ## then import model from LSTM.py below
    ## and also load model to device
    ## -----------------------------------------------
    model = LSTMModel(vocab_size=vocab_size,output_size=output_size,embedding_dim=embedding_dim,embedding_matrix=embedding_matrix,hidden_dim=hidden_dim,n_layers=n_layers,input_len=input_len)
    #model2 = CNN()
    model.to(device)
    ##-----------------------------------------------------------
    ## step 5: complete code to define optimizer and loss function
    ##-----------------------------------------------------------
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  # 
    ## define Binary Cross Entropy Loss below
    loss_fun = nn.BCELoss()
    
    ## step 6: load checkpoint
    epoches = 0
    if load_cpt:
        print("*"*10+'loading checkpoint'+'*'*10)
        ##-----------------------------------------------   
        ## complete code below to load checkpoint
        ##-----------------------------------------------
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoches = checkpoint['epoch']
    writer = SummaryWriter(log_dir='tensorboard_logs') #visualize loss over time


    ## step 7: model training
    print('*'*89)
    print('start model training now')
    print('*'*89)
    if mode == 'train':
        with open("log.txt", "w") as f: #reset the log
			# Write the loss and accuracy values to the file
            f.truncate(0)
            f.close()
        model.train()
        for epoch in range(epoches, num_epoches):
            adjust_learning_rate(learning_rate,optimizer,epoch)
            for batch_id, (x_batch,y_labels) in enumerate(training_generator):
                
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                ##-----------------------------------------------
                ## complete code to get predict result from model
                ##-----------------------------------------------
                y_out = model(x_batch)

                ##-----------------------------------------------
                ## complete code to get loss
                ##-----------------------------------------------
                loss = loss_fun(y_out, y_labels)

                ## step 8: back propagation [Done]
                optimizer.zero_grad()
                loss.backward()
                ## clip_grad_norm helps prevent the exploding gradient problem in LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                writer.add_scalar('training loss', loss.item(), epoch)
                with open("log.txt", "a") as f:
    				# Write the loss and accuracy values to the file
                    f.write("Epoch {}: Loss = {:.4f}\n".format(epoch,loss.item()))
                    f.close()
            ##-----------------------------------------------   
            ## step 9: complete code below to save checkpoint
            ##-----------------------------------------------
            #print("**** save checkpoint ****")
            _save_checkpoint(ckp_path,model,epoch,batch_id,optimizer) #Not entirely sure what global_step should be
    
    ##------------------------------------------------------------------
    ## step 10: complete code below for model testing
    ## predict result is a single value between 0 and 1, such as 0.8, so
    ## we can use y_pred = torch.round(y_out) to predict label 1 or 0
    ##------------------------------------------------------------------
    print("----model testing now----")
    model.eval()
    total = 0
    total_correct = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    with torch.no_grad():
        for batch_id, (x_batch,y_labels) in enumerate(test_generator):
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
            num_correct = (y_pred == y_labels).sum().item()
            true_pos += ((y_pred == y_labels) & (y_pred == 1)).sum().item()
            true_neg += ((y_pred == y_labels) & (y_pred == 0)).sum().item()
            false_pos += ((y_pred != y_labels) & (y_pred == 1)).sum().item()
            false_neg += ((y_pred != y_labels) & (y_pred == 0)).sum().item()
            # Compute the total number of predictions
            num_total = y_labels.size(0)
            total_correct += num_correct
            total += num_total
            # Compute the accuracy as the fraction of correct predictions
            #accuracy = num_correct / num_total * 100
            #print("Batch " + str(batch_id) + " Accuracy: " + str(accuracy))
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_1 = 2 * precision * recall / (precision + recall)
    print("Embedding Dimension: " + str(embedding_dim))
    print("Accuracy: " + str(total_correct / total))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("f_1: " + str(f_1))
    print("True Positives: " + str(true_pos))
    print("True Negatives: " + str(true_neg))
    print("False Positives: " + str(false_pos))
    print("False Negatives: " + str(false_neg))
    writer.close() #close tensorboard
    




if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    


    

    