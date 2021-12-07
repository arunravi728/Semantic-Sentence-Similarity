import re
import math
import pickle
import copy
import tqdm

from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

WINDOW_SIZE = 5             # WINDOW_SIZE number of words on either side of the input word
VOCABULARY_SIZE = 0
CORPUS_SIZE = 0

VOCABULARY = {}
UNIGRAM_RATIOS = []
BERNOULLI_MAP = []          # Bernoulli Map to help with subsampling

"""
Function to get string from a file
Input: File Name
Output: String
"""
def getText(fileName):
    file = open(fileName, "r")
    return file.read()

"""
Function to tokenize strings
Input: Text
Output: Tokenized version of string, Corpus Size
"""
def generateTokens(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

"""
Function to generate Vocabulary
Input: Tokens
Output: Vocabulary, Vocabulary Size
"""
def generateVocabulary(tokens):
    vocabulary = {}

    for i, token in tqdm.tqdm(enumerate(set(tokens))):
        vocabulary[token] = i
    
    return vocabulary, len(vocabulary)

"""
Function to generate Unigram Ratios
Input: Tokens
Output: Unigram Ratios
"""
def generateUnigramRatios(tokens):
    unigram_ratios = torch.zeros(VOCABULARY_SIZE)
    
    for token in tqdm.tqdm(tokens):
        unigram_ratios[VOCABULARY[token]] += 1
    
    return unigram_ratios/CORPUS_SIZE

"""
Function to generate Bernoulli Map to help with sub-sampling
Input: Tokens
Output: Bernoulli Map
"""
def generateBernoulliMap(tokens):
    bernoulli_map = copy.deepcopy(tokens)

    for i in tqdm.tqdm(range(len(tokens))):
        for j in range(len(tokens[i])):
            ratio = UNIGRAM_RATIOS[VOCABULARY[tokens[i][j]]]
            p_keep = (np.sqrt(ratio * 1000) + 1) * (0.001/ratio)
            if(ratio <= 0.0026):
                bernoulli_map[i][j] = 1
                continue
            bernoulli_map[i][j] = torch.from_numpy(np.random.binomial(1, p_keep, 1)).item()
    
    return bernoulli_map

"""
Function to generate One Hot Encoded Vector of a word
Input: word
Output: One Hot Encoded Vector
"""
def generateOneHotEncoding(word):
    one_hot_encoded_vector = torch.zeros(VOCABULARY_SIZE)
    one_hot_encoded_vector[VOCABULARY[word]] = 1
    return one_hot_encoded_vector

"""
Function to generate Training Data with Subsampling
Input: text
Output: Sub-Sampled Training Data
"""
def generateTrainingData(tokens):
    train_data = []
    for i, token in enumerate(tqdm.tqdm(tokens)):
        for j, t in enumerate(token):
            if(BERNOULLI_MAP[i][j] == 0):
                continue
            X = token[j]
            Y = []
            if(j < WINDOW_SIZE):
                for k in range(0,j):
                    if(BERNOULLI_MAP[i][k] == 1):
                        Y.append(tokens[i][k])
                for k in range(j+1, min(j + WINDOW_SIZE + 1,len(token) - 1)):
                    if(k != j):
                        if(BERNOULLI_MAP[i][k] == 1):
                            Y.append(tokens[i][k])
            elif(j + WINDOW_SIZE > len(token) - 1):
                for k in range(j - WINDOW_SIZE,j):
                    if(k != j):
                        if(BERNOULLI_MAP[i][k] == 1):
                            Y.append(tokens[i][k])
                for k in range(j+1, len(token)):
                    if(BERNOULLI_MAP[i][k] == 1):
                        Y.append(tokens[i][k])
            else:
                for k in range(int(j - WINDOW_SIZE), int(j + WINDOW_SIZE + 1)):
                    if(k != j):
                        if(BERNOULLI_MAP[i][k] == 1):
                            Y.append(tokens[i][k])   
            train_data.append(tuple((X,Y)))
    
    return train_data

text = getText("Data/data_word2vec.txt")
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

data_file = open("data_word2vec.pkl",'wb')
pickle.dump(sentences,data_file)
data_file.close()

data_file = open("Pickle/data_word2vec.pkl",'rb')
data = pickle.load(data_file)
data_file.close()

tokens_vocab = []
tokens_train = []

for sentence in tqdm.tqdm(sentences):
    tokens = generateTokens(sentence)
    tokens_train.append(tokens)
    for token in tokens:
        tokens_vocab.append(token)
        CORPUS_SIZE += 1

token_vocabulary_file = open("Pickle/token_vocabulary_file.pkl",'wb')
pickle.dump(tokens_vocab,token_vocabulary_file)
token_vocabulary_file.close()

token_train_file = open("Pickle/token_train_file.pkl",'wb')
pickle.dump(tokens_train,token_train_file)
token_train_file.close()

token_vocabulary_file = open("Pickle/token_vocabulary_file.pkl",'rb')
tokens_vocab = pickle.load(token_vocabulary_file)
token_vocabulary_file.close()

token_train_file = open("Pickle/token_train_file.pkl",'rb')
tokens_train = pickle.load(token_train_file)
token_train_file.close()

VOCABULARY, VOCABULARY_SIZE = generateVocabulary(tokens_vocab)
vocabulary_file = open("Pickle/vocabulary_file.pkl",'wb')
pickle.dump(VOCABULARY,vocabulary_file)
vocabulary_file.close()

vocabulary_file = open("Pickle/vocabulary_file.pkl",'rb')
VOCABULARY = pickle.load(vocabulary_file)
vocabulary_file.close()

CORPUS_SIZE = len(tokens_vocab)
VOCABULARY_SIZE = len(VOCABULARY)

UNIGRAM_RATIOS = generateUnigramRatios(tokens_vocab)
unigram_ratios_file = open("Pickle/unigram_ratios_file.pkl",'wb')
pickle.dump(UNIGRAM_RATIOS,unigram_ratios_file)
unigram_ratios_file.close()

unigram_file = open("Pickle/unigram_ratios_file.pkl",'rb')
UNIGRAM_RATIOS = pickle.load(unigram_file)
unigram_file.close()

BERNOULLI_MAP = generateBernoulliMap(tokens_train)
bernoulli_map_file = open("Pickle/bernoulli_map_file.pkl",'wb')
pickle.dump(BERNOULLI_MAP,bernoulli_map_file)
bernoulli_map_file.close()

bernoulli_map_file = open("Pickle/bernoulli_map_file.pkl",'rb')
BERNOULLI_MAP = pickle.load(bernoulli_map_file)
bernoulli_map_file.close()

train_data = generateTrainingData(tokens_train)
train_data_file = open("Pickle/train_data_file.pkl",'wb')
pickle.dump(train_data,train_data_file)
train_data_file.close()

train_data_file = open("Pickle/train_data_file.pkl",'rb')
train_data = pickle.load(train_data_file)
train_data_file.close()