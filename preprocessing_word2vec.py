import re

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
    return pattern.findall(text.lower()), len(pattern.findall(text.lower()))

"""
Function to generate Vocabulary
Input: Tokens
Output: Vocabulary, Vocabulary Size
"""
def generateVocabulary(tokens):
    vocabulary = {}
    
    for i, token in enumerate(set(tokens)):
        vocabulary[token] = i
    
    return vocabulary, len(vocabulary)

"""
Function to generate Unigram Ratios
Input: Tokens
Output: Unigram Ratios
"""
def generateUnigramRatios(tokens):
    unigram_ratios = torch.zeros(VOCABULARY_SIZE)
    
    for token in tokens:
        unigram_ratios[VOCABULARY[token]] += 1
    
    return unigram_ratios/CORPUS_SIZE

"""
Function to generate Bernoulli Map to help with sub-sampling
Input: Tokens
Output: Bernoulli Map
"""
def generateBernoulliMap(tokens):
    bernoulli_map = torch.ones(CORPUS_SIZE)

    for i in range(len(tokens)):
        ratio = UNIGRAM_RATIOS[VOCABULARY[tokens[i]]]
        p_keep = (np.sqrt(ratio * 1000) + 1) * (0.001/ratio)
        bernoulli_map[i] = np.random.choice(np.array([0,1]), p = [round(1 - p_keep.item(),3), round(p_keep.item(),3)])
    
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
    for i, token in enumerate(tokens):
        if(BERNOULLI_MAP[i] == 0):
            continue
        X = generateOneHotEncoding(token)
        if(i < WINDOW_SIZE):
            for j in range(0,i):
                if(BERNOULLI_MAP[j] == 1):
                    train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
            for j in range(i+1, i + WINDOW_SIZE + 1):
                if(j != i):
                    if(BERNOULLI_MAP[j] == 1):
                        train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
        elif(i + WINDOW_SIZE > len(tokens) - 1):
            for j in range(i - WINDOW_SIZE,i):
                if(j != i):
                    if(BERNOULLI_MAP[j] == 1):
                        train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
            for j in range(i+1, len(tokens)):
                if(BERNOULLI_MAP[j] == 1):
                    train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
        else:
            for j in range(int(i - WINDOW_SIZE), int(i + WINDOW_SIZE + 1)):
                if(j != i):
                    if(BERNOULLI_MAP[j] == 1):
                        train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
    return train_data

text = getText("data.txt")
tokens, CORPUS_SIZE = generateTokens(text)
VOCABULARY, VOCABULARY_SIZE = generateVocabulary(tokens)
UNIGRAM_RATIOS = generateUnigramRatios(tokens)
BERNOULLI_MAP = generateBernoulliMap(tokens)
train_data = generateTrainingData(tokens)