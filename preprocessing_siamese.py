import re
import math
import pickle
import tqdm

import warnings
warnings.filterwarnings("ignore")

from loguru import logger

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import torch
from transformers import MarianMTModel, MarianTokenizer

from gensim.models import Word2Vec, KeyedVectors

NUM_SAMPLES = 0
NUM_TRAIN = 0
NUM_TEST = 0
LAMBDA = np.random.beta(0.4,0.4)
EMBEDDING_LENGTH = 300

SENTENCES_1 = []
SENTENCES_2 = []
SIMILARITY_SCORES = []
STOPWORDS = []
WORD_EMBEDDINGS = []
OWN_EMBEDDINGS = []
VOCABULARY = []

#Model Variant (OWN_EMBEDDINGS, GOOGLE_NEWS, GENSIM_SKIP_GRAM)
MODE = "GENSIM_SKIP_GRAM"

"""
Function to tokenize strings
Input: Text
Output: Tokenized version of string, Corpus Size
"""
def generateTokens(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

"""
Function to parse data from SICK dataset
Input: Lines read from file
Output: Sentence 1, Sentence 2, Similarity Scores
"""
def parseData(lines):
    sentences_1 = []
    sentences_2 = []
    similarity_scores = []

    for line in lines:
        tab_count = 0
        s1 = ""
        s2 = ""
        score = ""
        s1_stop = False
        s2_stop = False

        for char in line:
            if(char == '\t'):
                tab_count += 1

            if(tab_count == 2 and char=='\t' and s1_stop == False):
                s1 += ". "
            elif(tab_count == 3 and char=='\t' and s2_stop == False):
                s2 += ". "

            if(tab_count == 1 and char!='\t'):
                s1 += str(char)
                if(char == '.'):
                    s1_stop = True
            elif(tab_count == 2 and char!='\t'):
                s2 += str(char)
                if(char == '.'):
                    s2_stop = True
            elif(tab_count == 4 and char!='\t'):
                score += str(char)
        
        sentences_1.append(s1)
        sentences_2.append(s2)
        similarity_scores.append(score)

    sentences_1.pop(0)
    sentences_2.pop(0)
    similarity_scores.pop(0)
    
    return sentences_1, sentences_2, similarity_scores

"""
Function to randomly delete a stop word from a sentence
Input: Sentence
Output: Sentence with a stop word deleted
"""
def deleteRandomStopWord(sentence):
    tokens = generateTokens(sentence)
    probabilities = np.ones(len(tokens))

    for i,token in enumerate(tokens):
        if token in STOPWORDS:
            probabilities[i] = np.random.uniform(0,1)
    
    words = []
    for i,token in enumerate(tokens):
        if i != np.argmin(probabilities):
            words.append(token)
    
    new_sentence = " ".join(words)
    new_sentence += "."
    return new_sentence

"""
Function to randomly insert a stop word in a sentence
Input: Sentence
Output: Sentence with a stop word inserted
"""
def insertRandomStopWord(sentence):
    tokens = generateTokens(sentence)
    probabilities = np.ones(len(STOPWORDS))

    for i,word in enumerate(STOPWORDS):
        probabilities[i] = np.random.uniform(0,1)

    stop_word = STOPWORDS[np.argmax(probabilities)]
    position = np.random.randint(0,len(tokens))
    words = []
    for i,token in enumerate(tokens):
        if i == position:
            words.append(stop_word)
            words.append(token)
        else:
            words.append(token)
    
    new_sentence = " ".join(words)
    new_sentence += "."
    return new_sentence

"""
Function to generate back translated sentence
Input: Sentence
Output: Back translated sentence
"""
def generateBackTranslatedSentence(sentence):
    source_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
    source_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')

    target_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
    target_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

    # translating from English to French
    encoded_sentence = target_tokenizer.prepare_seq2seq_batch([sentence],return_tensors = 'pt')
    translation = target_model.generate(**encoded_sentence)
    translated_sentence = target_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    # translating from French to English
    encoded_sentence = source_tokenizer.prepare_seq2seq_batch([translated_sentence],return_tensors = 'pt')
    back_translation = source_model.generate(**encoded_sentence)
    back_translated_sentence = source_tokenizer.batch_decode(back_translation, skip_special_tokens=True)[0]

    return back_translated_sentence

"""
Function to check if synonym exists in WordNet
Input: Word
Output: True/False
"""
def doesSynonymExist(word):
    exists = False
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            if(lemma.name().lower() != word.lower()):
                exists = True
    return exists

"""
Function to generate synonyms of a word from WordNet
Input: Word
Output: List of synonyms and similarity scores from WordNet
"""
def generateSynonyms(word):
    if(len(wordnet.synsets(word)) != 0):
        string = wordnet.synsets(word)[0]
    synonyms = []
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            if(lemma.name().lower() != word.lower() and lemma.name().lower() not in synonyms):
                synonyms.append(tuple((lemma.name().replace("_", " ").replace("-", " ").lower(),string.wup_similarity(synonym))))
    
    return synonyms

"""
Function to generate synonym replaced sentence
Input: Sentence
Output: Synonym replaced sentence sentence
"""
def generateSynonymReplacedSentence(sentence):
    tokens = generateTokens(sentence)
    max_similarity = 0
    max_similarity_index = 0
    synonym = ""

    for i,token in enumerate(tokens):
        if token not in STOPWORDS:
            if(doesSynonymExist(token) == True):
                synonyms = generateSynonyms(token)
                synonyms = sorted(synonyms, key = lambda x: x[1], reverse = True)
                if(synonyms[0][1] > max_similarity):
                    max_similarity_index = i
                    max_similarity = synonyms[0][1]
                    synonym = synonyms[0][0]
    
    words = []
    for i,token in enumerate(tokens):
        if i != max_similarity_index:
            words.append(token)
        else:
            words.append(synonym)
    
    new_sentence = " ".join(words)
    new_sentence += "."
    return new_sentence

"""
Function to generate a sentence through mixup
Input: Indices of two sentences to be mixedup
Output: Mixedup sentence in vector form
"""
def generateMixedupSentence(m, n):
    tokens_m = generateTokens(SENTENCES_1[m])
    tokens_n = generateTokens(SENTENCES_1[n])

    line = []

    new_sentence_1 = []
    for i in range(0,max(len(tokens_m),len(tokens_n))):
        if i < len(tokens_m):
            if MODE == "GENSIM_SKIP_GRAM":
                embedding_m = WORD_EMBEDDINGS.wv[tokens_m[i]]
            elif MODE == "OWN_EMBEDDINGS":
                embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
            elif MODE == "GOOGLE_NEWS":
                if tokens_m[i] in WORD_EMBEDDINGS:
                    embedding_m = torch.from_numpy(WORD_EMBEDDINGS[tokens_m[i]])
                else:
                    embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
        else:
            embedding_m = torch.zeros((300))

        if i < len(tokens_n):
            if MODE == "GENSIM_SKIP_GRAM":
                embedding_n = WORD_EMBEDDINGS.wv[tokens_n[i]]
            elif MODE == "OWN_EMBEDDINGS":
                embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
            elif MODE == "GOOGLE_NEWS":
                if tokens_n[i] in WORD_EMBEDDINGS:
                    embedding_n = torch.from_numpy(WORD_EMBEDDINGS[tokens_n[i]])
                else:
                    embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
        else:
            embedding_n = torch.zeros((300))

        new_sentence_1.append(LAMBDA*torch.tensor(embedding_m) + (1 - LAMBDA)*torch.tensor(embedding_n))

    line.append(new_sentence_1)

    tokens_m = generateTokens(SENTENCES_2[m])
    tokens_n = generateTokens(SENTENCES_2[n])

    new_sentence_2 = []
    for i in range(0,max(len(tokens_m),len(tokens_n))):
        if i < len(tokens_m):
            if MODE == "GENSIM_SKIP_GRAM":
                embedding_m = WORD_EMBEDDINGS.wv[tokens_m[i]]
            elif MODE == "OWN_EMBEDDINGS":
                embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
            elif MODE == "GOOGLE_NEWS":
                if tokens_m[i] in WORD_EMBEDDINGS:
                    embedding_m = torch.from_numpy(WORD_EMBEDDINGS[tokens_m[i]])
                else:
                    embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
        else:
            embedding_m = torch.zeros((300))

        if i < len(tokens_n):
            if MODE == "GENSIM_SKIP_GRAM":
                embedding_n = WORD_EMBEDDINGS.wv[tokens_n[i]]
            elif MODE == "OWN_EMBEDDINGS":
                embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
            elif MODE == "GOOGLE_NEWS":
                if tokens_n[i] in WORD_EMBEDDINGS:
                    embedding_n = torch.from_numpy(WORD_EMBEDDINGS[tokens_n[i]])
                else:
                    embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
        else:
            embedding_n = torch.zeros((300))

        new_sentence_2.append(LAMBDA*torch.tensor(embedding_m) + (1 - LAMBDA)*torch.tensor(embedding_n))

    line.append(new_sentence_2)

    score_m = SIMILARITY_SCORES[m]
    score_n = SIMILARITY_SCORES[n]
    new_score = LAMBDA*float(score_m) + (1 - LAMBDA)*float(score_n)
    line.append(new_score)

    return line

"""
Function to generate non-augmented Sick Dataset
"""
def generateBaseSickData():
    file = open("Data/data_sick.txt", "w")
    for i in range(0,NUM_SAMPLES):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("SICK Data Generated")

"""
Function to generate non-augmented Sick Dataset (Training)
"""
def generateBaseSickTrainData():
    file = open("Data/data_sick_train.txt", "w")
    for i in range(0,int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("SICK Train Data Generated")

"""
Function to generate non-augmented Sick Dataset (Test)
"""
def generateBaseSickTestData():
    file = open("Data/data_sick_test.txt", "w")
    for i in range(int(NUM_TRAIN),int(NUM_SAMPLES)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("SICK Test Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Random Stop Word Deletion
"""
def generateRandomDeletionData():
    file = open("Data/data_random_deletion.txt", "w")
    for i in range(0,int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = deleteRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = deleteRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("Random Deletion Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Random Stop Word Insertion
"""
def generateRandomInsertionData():
    file = open("Data/data_random_insertion.txt", "w")
    for i in range(0,int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = insertRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = insertRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("Random Insertion Data Generated")

"""
Function to generate Sick Dataset (Training) augmented  using Synonym Replacement
"""
def generateSynonymReplacementData():
    file = open("Data/data_synonym_replaced.txt", "w")
    for i in tqdm.tqdm(range(0,int(NUM_TRAIN))):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        line = SENTENCES_1[i] + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("Synonym Replacement Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Back Translation
"""
def generateBackTranslationData():
    file = open("Data/data_back_translation.txt", "w")
    for i in range(0,int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateBackTranslatedSentence(SENTENCES_1[i])
        new_sentence_2 = generateBackTranslatedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        line = SENTENCES_1[i] + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

    print("Back Translation Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Mixup
"""
def generateMixupData():
    lines = []
    for i in tqdm.tqdm(range(0,int(NUM_TRAIN))):
        random_indices = np.array([np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN))])

        lines.append(generateMixedupSentence(i,i))
        lines.append(generateMixedupSentence(i,random_indices[0]))
        lines.append(generateMixedupSentence(i,random_indices[1]))
        lines.append(generateMixedupSentence(i,random_indices[2]))

    if MODE == "GENSIM_SKIP_GRAM":
        mixedup_file = open("Data/GS_mixedup_file.pkl",'wb')
    elif MODE == "OWN_EMBEDDINGS":
        mixedup_file = open("Data/OE_mixedup_file.pkl",'wb')
    elif MODE == "GOOGLE_NEWS":
        mixedup_file = open("Data/GN_mixedup_file.pkl",'wb')
    
    pickle.dump(lines,mixedup_file)
    mixedup_file.close()

    print("Mixup Data Generated")

"""
Function to generate Word Embedded Sentences (Used for multiple data augmentations with Mixup)
"""
def generateWordEmbeddedSentences(sentence1, sentence2, score):
    tokens_m = generateTokens(sentence1)
    tokens_n = generateTokens(sentence2)

    line = []

    new_sentence_1 = []
    for i in range(0,len(tokens_m)):
        if MODE == "GENSIM_SKIP_GRAM":
            embedding_m = WORD_EMBEDDINGS.wv[tokens_m[i]]
        elif MODE == "OWN_EMBEDDINGS":
            embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
        elif MODE == "GOOGLE_NEWS":
            if tokens_m[i] in WORD_EMBEDDINGS:
                embedding_m = torch.from_numpy(WORD_EMBEDDINGS[tokens_m[i]])
            else:
                embedding_m = OWN_EMBEDDINGS[VOCABULARY[tokens_m[i]]]
        new_sentence_1.append(torch.tensor(embedding_m))

    line.append(new_sentence_1)

    new_sentence_2 = []
    for i in range(0,len(tokens_n)):
        if MODE == "GENSIM_SKIP_GRAM":
            embedding_n = WORD_EMBEDDINGS.wv[tokens_n[i]]
        elif MODE == "OWN_EMBEDDINGS":
            embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
        elif MODE == "GOOGLE_NEWS":
            if tokens_n[i] in WORD_EMBEDDINGS:
                embedding_n = torch.from_numpy(WORD_EMBEDDINGS[tokens_n[i]])
            else:
                embedding_n = OWN_EMBEDDINGS[VOCABULARY[tokens_n[i]]]
        new_sentence_2.append(torch.tensor(embedding_n))

    line.append(new_sentence_2)
    line.append(score)
    return line

"""
Function to generate Sick Dataset (Training) augmented using Mixup and Random Deletion
"""
def generateMxRd():
    lines = []

    for i in tqdm.tqdm(range(0, int(NUM_TRAIN/2))):
        random_indices = np.array([np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN))])

        lines.append(generateMixedupSentence(i,i))
        lines.append(generateMixedupSentence(i,random_indices[0]))
        lines.append(generateMixedupSentence(i,random_indices[1]))
        lines.append(generateMixedupSentence(i,random_indices[2]))
    
    for i in tqdm.tqdm(range(int(NUM_TRAIN/2), int(NUM_TRAIN))):
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], SENTENCES_2[i], SIMILARITY_SCORES[i]))               

        new_sentence = deleteRandomStopWord(SENTENCES_1[i])
        lines.append(generateWordEmbeddedSentences(new_sentence, SENTENCES_2[i], SIMILARITY_SCORES[i]))

        new_sentence = deleteRandomStopWord(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], new_sentence, SIMILARITY_SCORES[i]))  

        new_sentence_1 = deleteRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = deleteRandomStopWord(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(new_sentence_1, new_sentence_2, SIMILARITY_SCORES[i])) 

    if MODE == "GENSIM_SKIP_GRAM":
        mx_rd_file = open("Data/GS_MX_RD_file.pkl",'wb')
    elif MODE == "OWN_EMBEDDINGS":
        mx_rd_file = open("Data/OE_MX_RD_file.pkl",'wb')
    elif MODE == "GOOGLE_NEWS":
        mx_rd_file = open("Data/GN_MX_RD_file.pkl",'wb')
    pickle.dump(lines,mx_rd_file)
    mx_rd_file.close()

    print("MX - RD Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Mixup and Random Insertion
"""
def generateMxRi():
    lines = []

    for i in tqdm.tqdm(range(0, int(NUM_TRAIN/2))):
        random_indices = np.array([np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN))])

        lines.append(generateMixedupSentence(i,i))
        lines.append(generateMixedupSentence(i,random_indices[0]))
        lines.append(generateMixedupSentence(i,random_indices[1]))
        lines.append(generateMixedupSentence(i,random_indices[2]))
    
    for i in tqdm.tqdm(range(int(NUM_TRAIN/2), int(NUM_TRAIN))):
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], SENTENCES_2[i], SIMILARITY_SCORES[i]))               

        new_sentence = insertRandomStopWord(SENTENCES_1[i])
        lines.append(generateWordEmbeddedSentences(new_sentence, SENTENCES_2[i], SIMILARITY_SCORES[i]))

        new_sentence = insertRandomStopWord(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], new_sentence, SIMILARITY_SCORES[i]))  

        new_sentence_1 = insertRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = insertRandomStopWord(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(new_sentence_1, new_sentence_2, SIMILARITY_SCORES[i])) 

    mx_ri_file = open("Data/GS_MX_RI_file.pkl",'wb')
    pickle.dump(lines,mx_ri_file)
    mx_ri_file.close()

    print("MX - RI Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Mixup and Synonym Replacement
"""
def generateMxSr():
    lines = []

    for i in tqdm.tqdm(range(0, int(NUM_TRAIN/2))):
        random_indices = np.array([np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN)), np.random.randint(0,int(NUM_TRAIN))])

        lines.append(generateMixedupSentence(i,i))
        lines.append(generateMixedupSentence(i,random_indices[0]))
        lines.append(generateMixedupSentence(i,random_indices[1]))
        lines.append(generateMixedupSentence(i,random_indices[2]))
    
    for i in tqdm.tqdm(range(int(NUM_TRAIN/2), int(NUM_TRAIN))):
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], SENTENCES_2[i], SIMILARITY_SCORES[i]))               

        new_sentence = generateSynonymReplacedSentence(SENTENCES_1[i])
        lines.append(generateWordEmbeddedSentences(new_sentence, SENTENCES_2[i], SIMILARITY_SCORES[i]))

        new_sentence = generateSynonymReplacedSentence(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(SENTENCES_1[i], new_sentence, SIMILARITY_SCORES[i]))  

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        lines.append(generateWordEmbeddedSentences(new_sentence_1, new_sentence_2, SIMILARITY_SCORES[i])) 

    mx_sr_file = open("Data/GS_MX_SR_file.pkl",'wb')
    pickle.dump(lines,mx_sr_file)
    mx_sr_file.close()

    print("MX - SR Data Generated")

"""
Function to generate Sick Dataset (Training) augmented using Random Deletion and Random Insertion
"""
def generateRdRi():
    file = open("Data/RD_RI.txt", "w")
    for i in range(0,int(NUM_TRAIN/2)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = deleteRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = deleteRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    
    for i in range(int(NUM_TRAIN/2), int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = insertRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = insertRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

"""
Function to generate Sick Dataset (Training) augmented using Random Deletion and Synonym Replacement
"""
def generateRdSr():
    file = open("Data/RD_SR.txt", "w")
    for i in range(0,int(NUM_TRAIN/2)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = deleteRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = deleteRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = deleteRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    
    for i in range(int(NUM_TRAIN/2), int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        line = SENTENCES_1[i] + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

"""
Function to generate Sick Dataset (Training) augmented using Synonym Replacement and Random Insertion
"""
def generateSrRi():
    file = open("Data/SR_RI.txt", "w")
    for i in range(0,int(NUM_TRAIN/2)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        line = SENTENCES_1[i] + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = generateSynonymReplacedSentence(SENTENCES_1[i])
        new_sentence_2 = generateSynonymReplacedSentence(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    
    for i in range(int(NUM_TRAIN/2), int(NUM_TRAIN)):
        line = SENTENCES_1[i] + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_1[i])
        line = new_sentence + "\t" + SENTENCES_2[i] + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence = insertRandomStopWord(SENTENCES_2[i])
        line = SENTENCES_1[i] + "\t" + new_sentence + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())

        new_sentence_1 = insertRandomStopWord(SENTENCES_1[i])
        new_sentence_2 = insertRandomStopWord(SENTENCES_2[i])
        line = new_sentence_1 + "\t" + new_sentence_2 + "\t" + str(SIMILARITY_SCORES[i]) + "\n"
        file.write(line.lower())
    file.close()

file = open("Data/SICK.txt", "r")
lines = file.readlines()
file.close()

STOPWORDS = stopwords.words('english')
SENTENCES_1, SENTENCES_2, SIMILARITY_SCORES = parseData(lines)

NUM_SAMPLES = len(SENTENCES_1)
NUM_TRAIN = 0.8*NUM_SAMPLES
NUM_TEST = 0.2*NUM_SAMPLES

word_embeddings_file = open("Pickle/word_embeddings_file.pkl",'rb')
OWN_EMBEDDINGS = pickle.load(word_embeddings_file)
word_embeddings_file.close()

vocabulary_file = open("Pickle/vocabulary_file.pkl",'rb')
VOCABULARY = pickle.load(vocabulary_file)
vocabulary_file.close()

if MODE == "GENSIM_SKIP_GRAM":
    WORD_EMBEDDINGS = Word2Vec.load("Models/gensim_skipgram.model")
elif MODE == "GOOGLE_NEWS":
    WORD_EMBEDDINGS = KeyedVectors.load_word2vec_format('Models/GoogleNews-vectors-negative300.bin', binary = True)

# generateBaseSickData()
# generateBaseSickTrainData()
# generateBaseSickTestData()
# generateRandomDeletionData()
# generateRandomInsertionData()
# generateBackTranslationData()
# generateSynonymReplacementData()
# generateMixupData()
# generateMxRd()
# generateMxRi()
# generateMxSr()
# generateRdRi()
# generateRdSr()
# generateSrRi()