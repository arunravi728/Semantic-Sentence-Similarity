import torch
import torch
import torch.nn as nn
import math
import pickle
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim
from scipy import spatial
import numpy as np

#Re-implement in a cell later 
LONGEST_LENGTH = 0

#Depth of LSTM network
num_layers = 1

#Input Dimension
H_in = 300

#Dimension of hidden & cell states
H_hidden = 50

#Model Variant (OWN_EMBEDDINGS, GOOGLE_NEWS, GENSIM_SKIP_GRAM)
MODE = "OWN_EMBEDDINGS"

# #Reading word embeddings

word_embeddings_file = open("Pickle/word_embeddings_file.pkl",'rb')
WORD_EMBEDDINGS = pickle.load(word_embeddings_file)
word_embeddings_file.close()

#word2vec_model_google = KeyedVectors.load_word2vec_format('Models/GoogleNews-vectors-negative300.bin', binary = True)

#Reading vocabulary

vocabulary_file = open("Pickle/vocabulary_file.pkl",'rb')
VOCABULARY = pickle.load(vocabulary_file)
vocabulary_file.close()

#Loading custom word2vec model trained using gensim on SICK
#word2vec_model = Word2Vec.load("Models/gensim_skipgram.model")

word1 = "sea"
word2 = "fish"

vec1 = WORD_EMBEDDINGS[VOCABULARY[word1]]
vec2 = WORD_EMBEDDINGS[VOCABULARY[word2]]

print("OWN EMBEDDINGS - {}".format(1 - spatial.distance.cosine(vec1, vec2)))

#print(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))


vec1 = word2vec_model_google[word1]
vec2 = word2vec_model_google[word2]

print("Google News Dataset - {}".format(1 - spatial.distance.cosine(vec1, vec2)))

vec1 = word2vec_model.wv[word1]
vec2 = word2vec_model.wv[word2]

print("Gensim Skip Gram - {}".format(1 - spatial.distance.cosine(vec1, vec2)))

"""
Function to generate tokens
"""
def generateTokens(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

"""
Siamese Network
"""
class Siamese(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(Siamese, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #defining an LSTM with a depth of 1, not using a Stacked LSTM essentially, so we can use intermediate hidden states
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = num_layers, batch_first = True)

        #not defining an activation function here because the similarity fn auto scales to [0,1]

    """
    This a forward pass for every sentence through the Siamese Network
    """
    def forward_once(self, x):

        #Initializing hidden and cell states for every sequence (can switch to gaussian initalization here if needed)
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_dim)

        #pass input and (hidden_state, cell_state) here
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        #since num_layers = 1, out should store all intermediate hidden states and (hn,cn) will be a tuple containing final hidden state & cell states

        return hn
    
    """
    Forward pass the first sentence, then the second sentence
    """
    def forward(self, input1, input2):
        
        self.output1 = self.forward_once(input1)

        self.output2 = self.forward_once(input2)

        self.output1 = self.output1.view(-1, 50)

        self.output2 = self.output2.view(-1, 50)

        self.output = self.output1 - self.output2

        self.output3 = torch.abs(self.output)

        self.output4 = -1*torch.sum(self.output3, dim = 1)

        self.similarity_scores = torch.exp(self.output4)

        return self.similarity_scores

model = Siamese(H_in, H_hidden)

#Loading saved model

model.load_state_dict(torch.load("Models/OE_RD_TEST.pt"))

#Reading Test Set here for inference

f = open("Data/data_sick_test.txt", 'r')

#Variable to compute length of longest sequence

leng_long = 0

lines = f.readlines()

test_dataset_size = len(lines) #Should be 1968 ideally

s1_embeddings = []

s2_embeddings = []

labels = []

for line in lines:

  line = line.split("\t")

  sent1 = line[0]

  sent2 = line[1]

  labels.append(float(line[2]))

  sent1_embedding = []

  sent2_embedding = []

  for word in generateTokens(sent1):
    if MODE == "GOOGLE_NEWS":
      if word in word2vec_model:
          sent1_embedding.append(torch.from_numpy(word2vec_model[word]))
      else:
          sent1_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])
    elif MODE == "GENSIM_SKIP_GRAM":
      sent1_embedding.append(torch.from_numpy(word2vec_model.wv[word]))
    elif MODE == "OWN_EMBEDDINGS":
      sent1_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

  leng_long = len(generateTokens(sent1)) if len(generateTokens(sent1)) > leng_long else leng_long

  for word in generateTokens(sent2):
    if MODE == "GOOGLE_NEWS":
      if word in word2vec_model:
            sent2_embedding.append(torch.from_numpy(word2vec_model[word]))
      else:
            sent2_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])
    elif MODE == "GENSIM_SKIP_GRAM":
      sent2_embedding.append(torch.from_numpy(word2vec_model.wv[word]))
    elif MODE == "OWN_EMBEDDINGS":
      sent2_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

  leng_long = len(generateTokens(sent2)) if len(generateTokens(sent2)) > leng_long else leng_long

  s1_embeddings.append(sent1_embedding)

  s2_embeddings.append(sent2_embedding)

LONGEST_LENGTH = leng_long

labels = torch.FloatTensor(labels)

sent1 = "a cluster of four brown dogs are playing in a field of brown grass."

sent2 = "four dogs are playing in a area covered by grass."

#Retrieving word vectors from Gensim's pre-trained model on Google News Dataset

sent1_embedding = []

sent2_embedding = []

for word in generateTokens(sent1):
    sent1_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

for word in generateTokens(sent2):
    sent2_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

if len(sent1_embedding) < LONGEST_LENGTH:
    for _ in range(LONGEST_LENGTH - len(sent1_embedding)):
        sent1_embedding.append(torch.zeros(300))

if len(sent2_embedding) < LONGEST_LENGTH:
    for _ in range(LONGEST_LENGTH - len(sent2_embedding)):
        sent2_embedding.append(torch.zeros(300))

#Padding with zeros to match length of longest sequence

for idx in range(test_dataset_size):
  if len(s1_embeddings[idx]) < LONGEST_LENGTH:
      for _ in range(LONGEST_LENGTH - len(s1_embeddings[idx])):
          s1_embeddings[idx].append(torch.zeros(300))

  if len(s2_embeddings[idx]) < LONGEST_LENGTH:
      for _ in range(LONGEST_LENGTH - len(s2_embeddings[idx])):
          s2_embeddings[idx].append(torch.zeros(300))

for idx in range(test_dataset_size):
  s1_embeddings[idx] = torch.stack(s1_embeddings[idx])
  s2_embeddings[idx] = torch.stack(s2_embeddings[idx])

s1_embeddings = torch.stack(s1_embeddings)

s2_embeddings = torch.stack(s2_embeddings)

print(s1_embeddings.shape, s2_embeddings.shape)

#Reshaping to (batch_size, sequence_length, h_in) format to input to Siamese n/w 

sent1_embedding = torch.stack(sent1_embedding).reshape(1,LONGEST_LENGTH, H_in)

sent2_embedding = torch.stack(sent2_embedding).reshape(1,LONGEST_LENGTH, H_in)

#Scores outputted

scores = model.forward(sent1_embedding, sent2_embedding)

#Rescaling scores back to [0,5]
scores = scores * 5

print(scores)

#Measuring Test Error

criterion = nn.MSELoss()

print("Test Error is {}".format(criterion(scores, labels)))