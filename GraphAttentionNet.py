
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from random import shuffle


class GraphAttentionNet(nn.Module):
    def __init__(self, F, F_1, K, output_size):
        self.W = Variable(torch.rand(K, F_1, F), requires_grad=True)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.a = nn.Linear(2 * K * F_1, 1)
        self.h2o = nn.Linear(K * F_1, output_size)
        self.celu = nn.celu()
        self.K = K
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_u, input_n_u):
        u_h = torch.matmul(self.W, input_u)  # calculating the self attention outputs
        c_u_h = u_h.view(-1, 1)  # concatenating outputs of self attention layers

        v_h = torch.matmul(self.W, input_n_u)  # calculating the self attention outputs for neighbors
        c_v_h = v_h.view(-1, input_n_u.shape[1])  # concatenating the outputs of self attention layers

        W_v = torch.cat(tuple(torch.cat((c_u_h, x.view(-1, 1))) for x in torch.unbind(c_v_h, dim=1)), dim=1)
        alpha = self.a(W_v)
        alpha = self.leakyRelu(alpha)
        alpha = self.softmax(alpha)  # normalized attention coefficients

        u_h_1 = torch.sum(alpha * W_v)
        u_h_1 = self.celu(u_h_1)

        u_h_1 = u_h_1.view(-1, self.K)
        u_h_1 = torch.sum(u_h_1)
        output = self.h2o(u_h_1)
        output = self.sigmoid(output)

        return output


def getNodefeature(n,features):
    return torch.tensor(features[n],dtype=torch.float64).view(-1,1)

def getNodelabel(n,labels):
    return torch.tensor(labels[n],dtype=torch.long)

def getNeighborfeature(nbrs,features):
    nbr_f = getNodefeature(nbrs[0],features)
    for i in range(1,len(nbrs)):
        n_f = getNeighborfeature(nbrs[i],features)
        nbr_f = torch.cat((nbr_f,n_f),dim=1)
    return nbr_f

def testTrainSplit(node_cnt):
    train_size = int(0.8*node_cnt)
    nodes = [i for i in range(node_cnt)]
    shuffle(nodes)
    return nodes[:train_size],nodes[train_size:]


def train(graphAttnNet, train_nodes, features, labels, neighbors, epochs=1, learning_rate=0.001):
    optimizer = optim.Adam(graphAttnNet.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):

        for n in train_nodes:
            n_f = getNodefeature(n, features)
            n_l = getNodelabel(n, labels)
            nbr_f = getNeighborfeature(neighbors[n], features)

            out = graphAttnNet(n_f, nbr_f)
            loss = criterion(out, n_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__=="__main__":
    nodes = []
    node_map = {}
    neighbors = {}
    count = 0
    # get network structure
    with open('pygcn/data/cora/cora.cites') as fs:
        for line in fs:
            u, v = tuple(map(int, line.strip().split('\t')))
            if u not in nodes:
                nodes.append(u)
                node_map[u] = count
                neighbors[node_map[u]] = []
                count += 1
            if v not in nodes:
                nodes.append(v)
                node_map[v] = count
                neighbors[node_map[v]] = []
                count += 1

            neighbors[node_map[u]].append(node_map[v])
            neighbors[node_map[v]].append(node_map[u])

    # get features
    n_feature = {}
    n_class = {}
    cls_cnt = 0
    class_ = {}
    with open('pygcn/data/cora/cora.content') as fs:
        for line in fs:
            temp = line.strip().split('\t')
            node = node_map[int(temp[0])]
            n_feature[node] = temp[1:-1]
            cls = temp[-1]
            if cls not in class_:
                class_[cls] = cls_cnt
                cls_cnt += 1
            n_class[node] = class_[cls]

    feature_size = len(n_feature[0])
    F_1 = 8
    K = 8
    graphAttnNet = GraphAttentionNet(feature_size, F_1, K, len(class_))
    train_set, test_set = testTrainSplit(len(nodes))
    train(graphAttnNet, train_set, n_feature, n_class , neighbors)