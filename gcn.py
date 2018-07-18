# implements Graph Convolution Network proposed by Thomas Kiff et. al.
from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import networkx as nx
import numpy as np


def normalizedLaplacian(G):
    nodes = G.nodes()
    return nx.normalized_laplacian_matrix(G,nodes.sort())

def readGraph(fname):
    G = nx.Graph()
    with open(fname) as fs:
        for line in fs:
            u,v = line.strip().split()
            G.add_edge(int(u),int(v))
    return G

def readFeatureMatrix(fname):
    feat_vec = np.loadtxt(fname)
    return torch.tensor(feat_vec,dtype=torch.float64)

def readLabels(fname,nodes):
    labels = torch.zeros(nodes)
    i = 0
    with open(fname) as fs:
        ind = int(line.strip())
        labels[i] = ind
        i+=1
    return labels

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(GCN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.h2o = nn.Linear(input_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,norm_adj,feat_vec):
        input = torch.mm(norm_adj,feat_vec)
        hidden = self.i2h(input)
        hidden = torch.mm(norm_adj,hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output


def train(gcn,train_norm_adj,train_feat_vec,train_labels,iterations=1000,lr=0.001):
    optimizer = optim.SGD(vector.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    output = gcn(train_norm_adj, train_feat_vec)

    loss = criterion(output,train_labels)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

def evaluate(gcn,test_norm_adj,test_feat_vec,test_labels):

    output = gcn(test_norm_adj, test_feat_vec)
    value,index = torch.topk(output,1)
    for i in range(index.size[0]):
        print('✓' if index[i]==label[i] else '✗')


G = readGraph('input_network_file')
norm_adj = torch.tensor(normalizedLaplacian(G),dtype=torch.float64)
feat_vec = readFeatureMatrix('input_feature_file')
labels = readLabels('labels_file')
train_norm_adj,train_feat_vec,train_labels,test_norm_adj,test_feat_vec,test_labels = obtainTestTrainSplit(norm_adj,feat_vec,labels) # implement based on input data...
hidden_size = 20
input_size = feat_vec.shape[1]
output_size = 10 # number of output classes
gcn = GCN(input_size, hidden_size, output_size)
train(gcn,norm_adj,feat_vec)
