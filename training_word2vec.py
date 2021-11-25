import torch
import numpy as np
import pandas as pd
import math
import random
import tqdm

from preprocessing_word2vec import VOCABULARY, generateOneHotEncoding

#hidden dimensionality of the word embeddings
N = 300

#Vocabulary Size
V = len(VOCABULARY)

#Context Size
C = 5

#Read Vocabulary Here
VOCABULARY = {}

#Entire Corpus Stored Here
TOKENS = []

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
    unigram_counts = []

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
        self.W_in = (-2/math.sqrt(N))*torch.rand(V,N) + 1/math.sqrt(N)
        self.W_out = (-2/math.sqrt(N))*torch.rand(N,V) + 1/math.sqrt(N)

    """
    Function to perform a forward pass through the Skip Gram
    Input : Current Instance of Class
    Output : C probability distributions, each representing a context
    """
    def forward(self, input_vector):
        self.input_vector = input_vector
        self.h = torch.matmul(torch.transpose(self.W_in), self.input_vector)
        self.out = torch.matmul(torch.transpose(self.W_out), self.h)
        return self.out

    """
    Function to compute gradients wrt input & output vectors and the hidden layer output
    Input : Noise Distribution to sample negative examples from, 
    Output : All the gradients
    """
    def gradients(self, noise_distribution, input, input_encoded, context):

        #summing up & updating gradients for input vector over entire context

        '''if idx == 0:
            self.inp_grad = torch.zeros(N,1)
        
        elif VOCABULARY[prev_input] == VOCABULARY[input]:
            self.W_in[VOCABULARY[prev_input]] -= alpha * self.inp_grad
            self.inp_grad = torch.zeros(N,1)'''

        self.inp_grad = torch.zeros(N,1)
        
        for id in range(len(context)):

            ##performing the forward pass through the Skip Gram Network

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
            
            self.inp_grad += (torch.sigmoid(torch.dot(self.W_out[:,VOCABULARY[context]], self.h)) - 1)*self.W_out[:,VOCABULARY[context]]
        
        self.W_in[VOCABULARY[input]] -= alpha * self.inp_grad




"""
(input_word, list containing all context words in the window)
Read this variable from a txt file
"""
training_data = []

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