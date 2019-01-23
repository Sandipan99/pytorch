# implements a simple transfer learning setup.. we obtain a trained model on a simple classification task and then
# add a layer to the trained model to to work on a similar task but on a different data but only learn the parameters
#  for the new layer...

__author__ = "Sandipan Sikdar"

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from torch.distributions import multivariate_normal as mn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_data(samp_size,mean_1,covar_1,mean_2,covar_2): # generate synthetic data with a multivariate normal
    # distribution
    mean_1 = torch.Tensor(mean_1)
    covar_1 = torch.Tensor(covar_1)
    sampler_1 = mn.MultivariateNormal(mean_1,covar_1)

    mean_2 = torch.Tensor(mean_2)
    covar_2 = torch.Tensor(covar_2)
    sampler_2 = mn.MultivariateNormal(mean_2,covar_2)

    sample_lebel_1 = sampler_1.sample(sample_shape=(int(samp_size/2),1))
    sample_lebel_2 = sampler_2.sample(sample_shape=(int(samp_size/2),1))

    return sample_lebel_1,sample_lebel_2


def createTrainset(sample_label_1, sample_label_2,size):
    data = []
    labels = []
    cnt_samp_1 = 0
    cnt_samp_2 = 0
    sample = size//2

    while cnt_samp_1 < sample or cnt_samp_2 < sample:

        # selecting one of the classes randomly
        if random.random() > 0.5:
            if cnt_samp_1 < sample:
                input_tensor = sample_lebel_1[cnt_samp_1]
                label_tensor = torch.tensor(1, dtype=torch.float32).view(1, 1)
                cnt_samp_1 += 1
            else:
                input_tensor = sample_lebel_2[cnt_samp_2]
                label_tensor = torch.tensor(0, dtype=torch.float32).view(1, 1)
                cnt_samp_2 += 1
        else:
            if cnt_samp_2 < sample:
                input_tensor = sample_lebel_2[cnt_samp_2]
                label_tensor = torch.tensor(0, dtype=torch.float32).view(1, 1)
                cnt_samp_2 += 1
            else:
                input_tensor = sample_lebel_1[cnt_samp_1]
                label_tensor = torch.tensor(1, dtype=torch.float32).view(1, 1)
                cnt_samp_1 += 1

        data.append(input_tensor)
        labels.append(label_tensor)

    return data, labels

class FeedForward(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.LogSigmoid()

    def forward(self,input):
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.sigmoid(output)
        return output

def train(feedforward, input_data, input_labels, optimizer, epochs=1):
    for e in range(epochs):

        for i in range(len(input_data)):
            input_tensor = input_data[i]
            label_tensor = input_labels[i]

            input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

            output = feedforward(input_tensor)

            loss = torch.abs(output - label_tensor)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


class FeedForwardAdapt(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(FeedForwardAdapt, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size,hidden_size)
        for p in self.i2h.parameters():
            p.requires_grad = False
        self.h2h = nn.Linear(hidden_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)
        #for p in self.h2o.parameters():
        #    p.requires_grad = False
        self.sigmoid = nn.LogSigmoid()

    def forward(self,input):
        hidden = self.i2h(input)
        hidden = self.h2h(hidden)
        output = self.h2o(hidden)
        output = self.sigmoid(output)
        return output



if __name__=="__main__":
    input_size = 2
    hidden_size = 10
    output_size = 1
    learning_rate=0.001
    sample_lebel_1, sample_lebel_2 = generate_data(10000, [4, 4], [[2, 1], [1, 2]], [-4, -4], [[2, 1], [1, 2]])

    feedforward = FeedForward(input_size, hidden_size, output_size)
    feedforward.to(device)
    optimizer = optim.Adam(feedforward.parameters(), lr=learning_rate)
    input,labels = createTrainset(sample_lebel_1,sample_lebel_2,10000)
    train(feedforward, input, labels,optimizer)

    feedforwardnew = FeedForwardAdapt(input_size, hidden_size, output_size)

    params1 = feedforward.named_parameters()
    params2 = feedforwardnew.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data)

    optimizerNew = optim.Adam(filter(lambda x: x.requires_grad, feedforwardnew.parameters()), lr=learning_rate)
    sample_lebel_1_1, sample_lebel_2_1 = generate_data(1000, [2, 6], [[2, 1], [1, 2]], [-2, -6], [[2, 1], [1, 2]]) # new
    #  data
    train_data, train_labels = createTrainset(sample_lebel_1_1, sample_lebel_2_1, 1000)
    train(feedforwardnew,train_data,train_labels,optimizerNew)

    print(feedforward.i2h.bias)
    print(feedforward.i2h.weight)

    print(feedforward.h2o.bias)
    print(feedforward.h2o.weight)

    print('--------------------------------')

    print(feedforwardnew.i2h.bias)
    print(feedforwardnew.i2h.weight)

    print(feedforwardnew.h2o.bias)
    print(feedforwardnew.h2o.weight)

    print(feedforwardnew.h2h.bias)
    print(feedforwardnew.h2h.weight)
