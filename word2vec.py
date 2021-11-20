import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

CONTEXT_SIZE = 4
WINDOW_SIZE = 5           # WINDOW_SIZE number of words on either side of the input word
VOCABULARY_SIZE = 0

VOCABULARY = {}

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
Input: text
Output: Tokenized version of string
"""
def generateTokens(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

"""
Function to generate Vocabulary
Input: Tokens
Output: vocabulary, size of vocabulary
"""
def generateVocabulary(tokens):
    vocabulary = {}
    
    for i, token in enumerate(set(tokens)):
        vocabulary[token] = i
    
    return vocabulary, len(vocabulary)

"""
Function to generate One Hot Encoded Vector of a word
Input: word, vocabulary, size of vocabulary
Output: List of One Hot Encoded Vector
"""
def generateOneHotEncoding(word):
    one_hot_encoded_vector = torch.zeros(VOCABULARY_SIZE)
    one_hot_encoded_vector[VOCABULARY[word]] = 1
    return one_hot_encoded_vector

"""
Function to generate Training Data
Input: text
Output: Training Data -> List of One Hot Encoded Tuples
"""
def generateTrainingData(tokens):
    train_data = []
    for i, token in enumerate(tokens):
        X = generateOneHotEncoding(token)
        if(i < WINDOW_SIZE):
            for j in range(0,i):
                train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
            for j in range(i+1, i + WINDOW_SIZE + 1):
                if(j != i):
                    train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
        elif(i + WINDOW_SIZE > len(tokens) - 1):
            for j in range(i - WINDOW_SIZE,i):
                if(j != i):
                    train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
            for j in range(i+1, len(tokens)):
                train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
        else:
            for j in range(int(i - WINDOW_SIZE), int(i + WINDOW_SIZE + 1)):
                if(j != i):
                    train_data.append(tuple([X,generateOneHotEncoding(tokens[j])]))
    return train_data

text = getText("data.txt")
tokens = generateTokens(text)
VOCABULARY, VOCABULARY_SIZE = generateVocabulary(tokens)
train_data = generateTrainingData(tokens)
print(len(train_data))
