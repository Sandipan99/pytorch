
__author__ = "Sandipan Sikdar"

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from torch.distributions import multivariate_normal as mn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def generate_data(samp_size):
    mean_1 = torch.Tensor([4,4])
    covar_1 = torch.Tensor([[2,1],[1,2]])
    sampler_1 = mn.MultivariateNormal(mean_1,covar_1)

    mean_2 = torch.Tensor([-4,-4])
    covar_2 = torch.Tensor([[2,1],[1,2]])
    sampler_2 = mn.MultivariateNormal(mean_2,covar_2)

    sample_lebel_1 = sampler_1.sample(sample_shape=(int(samp_size/2),1))
    sample_lebel_2 = sampler_2.sample(sample_shape=(int(samp_size/2),1))

    return sample_lebel_1,sample_lebel_2

def train(feedforward, sample_lebel_1, sample_lebel_2, epochs=1, learning_rate=0.001):
    optimizer = optim.Adam(feedforward.parameters(), lr=learning_rate)
    cnt_samp_1 = 0
    cnt_samp_2 = 0
    iters = 0
    for i in range(epochs):
        while cnt_samp_1<5000 or cnt_samp_2<5000:
            iters+=1
            # selecting one of the classes randomly
            if random.random()>0.5:
                if cnt_samp_1<5000:
                    input_tensor = sample_lebel_1[cnt_samp_1]
                    label_tensor = torch.tensor(1,dtype=torch.float32).view(1,1)
                    cnt_samp_1+=1
                else:
                    input_tensor = sample_lebel_2[cnt_samp_2]
                    label_tensor = torch.tensor(0,dtype=torch.float32).view(1,1)
                    cnt_samp_2+=1
            else:
                if cnt_samp_2<5000:
                    input_tensor = sample_lebel_2[cnt_samp_2]
                    label_tensor = torch.tensor(0,dtype=torch.float32).view(1,1)
                    cnt_samp_2+=1
                else:
                    input_tensor = sample_lebel_1[cnt_samp_1]
                    label_tensor = torch.tensor(1,dtype=torch.float32).view(1,1)
                    cnt_samp_1+=1

            input_tensor, label_tensor = input_tensor.to(device),label_tensor.to(device)        

            output = feedforward(input_tensor)

            loss = torch.abs(output - label_tensor)

            if iters%100==0:
                print("Loss: ",loss)

            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()



if __name__=="__main__":
    input_size = 2
    hidden_size = 10
    output_size = 1
    sample_lebel_1,sample_lebel_2 = generate_data(10000)
    feedforward = FeedForward(2,10,1)
    feedforward.to(device)
    train(feedforward,sample_lebel_1,sample_lebel_2)

    print(feedforward.i2h.bias)
    print(feedforward.i2h.weight)

    print(feedforward.h2o.bias)
    print(feedforward.h2o.weight)
