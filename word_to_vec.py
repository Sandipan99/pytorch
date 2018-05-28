# implements word2vec with stochastic gradient descent... without negative sampling/hierarchical softmax..

__author__ = "Sandipan Sikdar" 

from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def tokenise(dir): # returns word2index, index2word
    translator = str.maketrans('', '', string.punctuation)
    word2index = {}
    sentence = []
    index = 0
    for root,dir,files in os.walk(dir):
        for f in files:
            fs = open(os.path.join(root,f))
            for line in fs:
                line = line.translate(translator)
                temp = line.strip().split(' ')
                for word in temp:
                    word.lower()
                    if word not in word2index:
                        word2index[word] = index
                        index+=1

    index2word = {}
    for word,index in word2index.items():
        index2word[index] = word

    return word2index, index2word

def valid(lst,i,j):
    l = len(lst)-1
    if (i-j)>=0 and (i+j)<=l:
        return 1
    elif (i+j)<=l:
        return 2
    else:
        return 3

def generateWordContextPair(dir, s_g):
    w2i, i2w = tokenise(dir)
    translator = str.maketrans('', '', string.punctuation)
    w_c_pairs = []
    sentences = []
    for root,dir,files in os.walk(dir):
        for f in files:
            fs = open(os.path.join(root,f))
            for line in fs:
                temp = line.strip().split('.')
                del temp[-1]
                temp = [s.strip().translate(translator) for s in temp]
                for s in temp:
                    words = s.split(' ')
                    words = [w2i[w.lower()] for w in words ]
                    sentences.append(words)


    for s in sentences:
        for i in range(len(s)):
            for j in range(1,s_g):
                if valid(s,i,j)==1:
                    w_c_pairs.append([s[i],s[i+j]])
                    w_c_pairs.append([s[i],s[i-j]])
                elif valid(s,i,j)==2:
                    w_c_pairs.append([s[i],s[i+j]])
                else:
                    w_c_pairs.append([s[i],s[i-j]])

    return w_c_pairs

w_c_pairs = generateWordContextPair("docs",2)

class to_vector(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(to_vector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input):
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output,hidden


def oneHotEncoding(w2i,w):
    x = torch.zeros(1,len(w2i), dtype=torch.float32)
    x[0][w] = 1
    return x

def obtainRandomPair(Pairs):
    x = random.randint(0,len(Pairs)-1)
    return Pairs[x]

def train(vector, w2i, iter = 5000, learning_rate = 0.01):
    vector_optimizer = optim.SGD(vector.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()


    for i in range(iter):
        pair = obtainRandomPair(w_c_pairs)
        input_tensor = oneHotEncoding(w2i,pair[0])
        target_tensor = torch.LongTensor([pair[1]])
        output, vector_hidden = vector(input_tensor)
        loss = criterion(output, target_tensor)

        vector_optimizer.zero_grad()

        loss.backward()
        vector_optimizer.step()

def obtainVectors(vector,w2i):
    for word,index in w2i.items():
        input_tensor = oneHotEncoding(w2i,index)
        output, hidden = vector(input_tensor)
        print (word,hidden)

hidden_size = 20
w2i,i2w = tokenise("docs")
input_size = len(w2i)
output_size = len(w2i)
vector = to_vector(input_size, hidden_size, output_size)
train(vector,w2i)
obtainVectors(vector,w2i)
