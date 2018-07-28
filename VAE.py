
# generate new words with character-level variational autoencoder (VAE)

__author__ = "Sandipan Sikdar"

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import string
import random

def obtainWords(fname): # parse files to obtain words removing punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    all_words = []
    with open(fname) as fs:
        for line in fs:
             words = line.translate(translator).strip().split()
             for w in words:
                 all_words.append(w)
    return all_words

def encodeWord(word,all_letters,n_letters):
    w = [all_letters.find(l) for l in word]
    word_tensor = torch.zeros(len(w)+1, 1, n_letters+1)
    for ind,letter in enumerate(w):
        word_tensor[ind][0][letter] = 1
    word_tensor[len(w)][0][n_letters] = 1
    return word_tensor


class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.std = nn.Linear(hidden_size, output_size)

    def forward(self,input,hidden):
        for ei in range(input.size(0)):
            embedded = self.embedding(input[ei]).view(1,1,-1) # view is same as reshape
            output = embedded
            output, hidden = self.gru(output, hidden)
        output_mean = mean(output)
        output_std = std(output)
        return output_mean,output_std

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,input,hidden):
        output = input
        output, hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size)

def obtainTrainingExample(all_words,all_letters,n_letters):
    word = all_words[random.randint(0,len(all_words)-1)]
    return encodeWord(word,all_letters,n_letters)



def train(encoder,decoder,normal_sampler,all_words,all_letters,n_letters,iterations=5000,learning_rate=0.001):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimozer = optim.Adam(decoder.parameters(), lr=learning_rate)

    for i in range(iterations):
        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_hidden = encoder.initHidden()
        decoder_hidden = decoder.initHidden()

        input_tensor = obtainTrainingExample(all_words,all_letters,n_letters)
        output_mean, output_std = encoder(input_tensor,encoder_hidden)

        # sample from standard normal with mean 0 and standard deviation 1
        output_std = torch.exp(output_std)

        for i in range(input_tensor.size(0)):
            norm = normal_sampler.sample(torch.tensor([output_mean.size(0)])).view(1,-1)
            decoder_input = torch.add(output_mean,torch.mul(output_std,norm))
            output,decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss = loss + torch.dist(input_tensor[i],output) - (torch.mul(torch.norm(output_mean),torch.norm(output_mean)) + torch.mul(torch.norm(output_std),torch.norm(output_std))-torch.sum(output_std)-output_std.size(1))*0.5

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

def generate(decoder,normal_sampler,all_letters,MAX_LENGTH=5):

    decoder_hidden = decoder.initHidden()
    word=''
    for i in range(MAX_LENGTH):
        norm = normal_sampler.sample(torch.tensor([output_mean.size(0)])).view(1,-1)
        decoder_input = torch.add(output_mean,torch.mul(output_std,norm))
        output,decoder_hidden = decoder(decoder_input, decoder_hidden)

        value,index = torch.topk(output,1)
        if index.item()==len(all_letters):
            break
        else:
            word+=all_letters[index.item()]

    print(word)


if __name__=="__main__":
    all_letters = string.ascii_letters
    n_letters = len(all_letters)
    all_words = obtainWords("Data/sample_male")
    normal_sampler = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    input_size = n_letters+1
    enc_hidden_size,dec_hidden_size = 20,20
    enc_output_size = 10
    encoder = Encoder(n_letters+1,enc_hidden_size,enc_output_size)
    decoder = Decoder(enc_output_size,dec_hidden_size,n_letters+1)
    train(encoder,decoder,normal_sampler,all_words,all_letters,n_letters)

    generate(decoder,normal_sampler,all_letters)
    #print(encodeWord(all_words[8],all_letters,n_letters))
