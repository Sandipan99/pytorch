# implements GLoVe with stochastic gradient descent... proposed by Pennington et. al.

__author__ = "Sandipan Sikdar"

import string
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
from torch import optim

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def obtainKFrequentWords(fname,k): # obtain the word frequencies and return the top k words

    vocab = Counter()
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    ps = PorterStemmer()

    doc = []

    with open(fname) as fs:
        for line in fs:
            line = line.strip().lower().translate(translator)

            for word in word_tokenize(line):
                word = ps.stem(word)
                if word not in stop_words:
                    vocab[word]+=1

                doc.append(word)

    return vocab.most_common(k),doc


def obtainFrequencyMatrix(fname,k,window_size):

    freqWords, doc = obtainKFrequentWords(fname,k)

    word2index = {}
    index2word = {}

    count = 0

    word_cooccur = np.zeros(k,k)

    for a,b in freqWords:
        word2index[a] = count
        index2word[count] = a
        count+=1

    curr_pos = 0
    word = doc[curr_pos]
    while word not in word2index:
        curr_pos+=1
        word = doc[curr_pos]

    while curr_pos<len(doc):

        m_word = doc[curr_pos]
        flag = 0
        for i in range(curr_pos+1,curr_pos+window_size):
            c_word = doc[i]
            if c_word in word2index:
                word_cooccur[word2index[m_word],word2index[c_word]]+=1/(i-curr_pos)
                word_cooccur[word2index[c_word],word2index[m_word]]+=1/(i-curr_pos)
                if flag==0:
                    curr_pos = i
                    flag = 1

    return word_cooccur, word2index, index2word


def matrix2tuple(word_cooccur):

    data = []

    x,y = word_cooccur.shape

    for i in range(x-1):
        for j in range(i+1,x):
            if word_cooccur[i,j]>0:
                data.append((i,j,word_cooccur[i,j]))
            if word_cooccur[j,i]>0:
                data.append((j,i,word_cooccur[j,i]))

    return data


class Glove(nn):
    def __init__(self, vocab_size, embedding_size, x_max, alpha):
        super(Glove, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.m_W = nn.Parameter(vocab_size, embedding_size)
        self.c_W = nn.Parameter(vocab_size, embedding_size)

        self.bias_m = nn.Parameter(vocab_size)
        self.bias_c = nn.Parameter(vocab_size)

        self.x_max = x_max
        self.alpha = alpha

    def forward(self, main_w, context_w, x_ij):

        f_x = (x_ij/self.x_max)**self.alpha if x_ij<self.x_max else 1

        output = torch.dot(self.m_W[main_w],self.c_W[context_w]) + bias_m[main_w] + bias_c[context_w] - torch.log(x_ij)
        output = output**2
        output = output*f_x

        return output

def train(glove,train_data,epochs=25,learning_rate=0.05):

    optimizer = optim.Adagrad(glove.parameters(), lr=learning_rate)

    for _ in range(epochs):

        shuffle(train_data)

        for a,b,x_ij in train_data:

            loss = glove(a, b, x_ij)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def getEmbedding(glove,index2word):

    embed_vec = glove.c_W + glove.m_W

    embedding = {}

    for i in range(len(index2word)):
        embedding[index2word[i]] = embed_vec[i]

    return embedding


if __name__=="__main__":

    word_cooccur,word2index,index2word = obtainFrequencyMatrix('samlpe_reviews',40000,10)
    train_data = matrix2tuple(word_cooccur)

    embedding_size = 300
    vocab_size = word_cooccur.shape[0]
    x_max = 100
    alpha = 0.75

    glove = Glove(vocab_size, embedding_size, x_max, alpha)

    train(glove, train_data)

    embedding = getEmbedding(glove,index2word)