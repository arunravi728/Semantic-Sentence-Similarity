import torch
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm

from preprocessing_word2vec import VOCABULARY, VOCABULARY_SIZE, tokens, train_data, UNIGRAM_RATIOS, generateOneHotEncoding

#hidden dimensionality of the word embeddings
N = 300


#Context Size
C = 5

#Read Vocabulary Here
#VOCABULARY = {}

#Entire Corpus Stored Here
TOKENS = tokens

#Learning Rate for SGD
alpha = 0.001

#No. of Negative Samples
K = 5

#Number of Epochs
num_epochs = 3

"""
Function to generate Noise Distribution (Needs to be executed exactly once!)
Input : Corpus Vocabulary
Output : Noise distribution for Negative Sampling
"""
def generateNoiseDist():
    #Read this variable from a txt file here
    unigram_counts = [ele*len(TOKENS) for ele in UNIGRAM_RATIOS]

    unigram_counts = [ele ** 0.75 for ele in unigram_counts]
    sum_counts = sum(unigram_counts)

    noise_distribution = [ele / sum_counts for ele in unigram_counts]

    return noise_distribution




"""Class implementing the Skip-Gram Model from scratch"""

class SkipGram():
    """
    Constructor initializes input and output weight matrices
    """
    def __init__(self):
        self.W_in = (-2/math.sqrt(N))*torch.rand(VOCABULARY_SIZE,N) + 1/math.sqrt(N)
        self.W_out = (-2/math.sqrt(N))*torch.rand(N,VOCABULARY_SIZE) + 1/math.sqrt(N)

    """
    Function to perform a forward pass through the Skip Gram
    Input : Current Instance of Class
    Output : C probability distributions, each representing a context
    """
    def forward(self, input_vector):
        self.input_vector = input_vector
        self.h = torch.matmul(torch.transpose(self.W_in, 0, 1), self.input_vector)
        self.out = torch.matmul(torch.transpose(self.W_out, 0 ,1), self.h)
        return self.out

    """
    Function to compute gradients wrt input & output vectors and the hidden layer output
    Input : Noise Distribution, input string, one hot encoded input, context string
    Output : All the gradients
    """
    def gradients(self, noise_distribution, input, input_encoded, context):

        self.inp_grad = torch.zeros(N)
        
        for id in range(len(context)):

            #performing the forward pass through the Skip Gram Network

            self.forward(input_encoded)

            #sampling K negative samples from the Noise Distribution
        
            D_dash = random.choices(list(VOCABULARY.keys()), noise_distribution, k = K)

            #updating output vectors of negative samples

            for i in range(K):
                self.W_out[:,VOCABULARY[D_dash[i]]] -= alpha * torch.sigmoid(torch.dot(self.W_out[:,VOCABULARY[D_dash[i]]], self.h))*self.h
                
            #updating output vector of the positive sample (our context word)

            self.W_out[:,VOCABULARY[context[id]]] -= alpha * (torch.sigmoid(torch.dot(self.W_out[:,VOCABULARY[context[id]]], self.h)) - 1)*self.h

            #computing gradient for input vector

            for i in range(K):
                self.inp_grad += torch.sigmoid(torch.dot(self.W_out[:,VOCABULARY[D_dash[i]]], self.h))* self.W_out[:,VOCABULARY[D_dash[i]]]
            
            self.inp_grad += (torch.sigmoid(torch.dot(self.W_out[:,VOCABULARY[context[id]]], self.h)) - 1)*self.W_out[:,VOCABULARY[context[id]]] 

        #updating the input vector after all contexts are done
        self.W_in[VOCABULARY[input]] -= alpha * self.inp_grad




"""
(input_word, list containing all context words in the window)
"""
training_data = train_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Generates the special unigram frequencies from which to negative sample from
noise_distribution = generateNoiseDist()

skip_gram = SkipGram()


for epoch in tqdm(range(num_epochs)):
    for idx, (input, context) in enumerate(training_data):

        input_encoded = generateOneHotEncoding(input).to(device)
        
        #context = generateOneHotEncoding(context).to(device)

        #skip_gram.forward(input_encoded)

        skip_gram.gradients(noise_distribution, input, input_encoded, context)

print(list(VOCABULARY.keys())[0])

print(skip_gram.W_in[0])

print(skip_gram.W_in[0].shape)