import torch
import torch
import torch.nn as nn
import math
import pickle
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#Re-implement this function here 
LONGEST_LENGTH = 32

#Depth of LSTM network
num_layers = 1

#Input Dimension
H_in = 300

#Dimension of hidden & cell states
H_hidden = 50

#Reading word embeddings

'''word_embeddings_file = open("Pickle/word_embeddings.pkl",'rb')
WORD_EMBEDDINGS = pickle.load(word_embeddings_file)
word_embeddings_file.close()

word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)'''

#Loading custom word2vec model trained using gensim on SICK
word2vec_model = Word2Vec.load("word2vec.model")

#Reading vocabulary

'''vocabulary_file = open("Pickle/vocabulary_file.pkl",'rb')
VOCABULARY = pickle.load(vocabulary_file)
vocabulary_file.close()'''

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

model.load_state_dict(torch.load("model_word2vec_custom.pt"))

sent1 = "A parrot is speaking"

sent2 = "The parrot is silent in front of the microphone"

#Retrieving word vectors from Gensim's pre-trained model on Google News Dataset

'''sent1_embedding = []

sent2_embedding = []

for word in generateTokens(sent1):
    if word in word2vec_model:
        sent1_embedding.append(torch.from_numpy(word2vec_model[word]))
    else:
        sent1_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

for word in generateTokens(sent2):
    if word in word2vec_model:
        sent2_embedding.append(torch.from_numpy(word2vec_model[word]))
    else:
        sent2_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])'''

#Retrieving word vectors from custom word2vec model trained using Gensim on SICK

sent1_embedding = []

sent2_embedding = []

for word in generateTokens(sent1):
    sent1_embedding.append(torch.from_numpy(word2vec_model.wv[word]))

for word in generateTokens(sent2):
    sent2_embedding.append(torch.from_numpy(word2vec_model.wv[word]))

#Padding with zeros to match length of longest sequence

if len(sent1_embedding) < LONGEST_LENGTH:
    for _ in range(LONGEST_LENGTH - len(sent1_embedding)):
        sent1_embedding.append(torch.zeros(300))

if len(sent2_embedding) < LONGEST_LENGTH:
    for _ in range(LONGEST_LENGTH - len(sent2_embedding)):
        sent2_embedding.append(torch.zeros(300))

#Reshaping to (batch_size, sequence_length, h_in) format to input to Siamese n/w 

sent1_embedding = torch.stack(sent1_embedding).reshape(1,LONGEST_LENGTH, H_in)

sent2_embedding = torch.stack(sent2_embedding).reshape(1,LONGEST_LENGTH, H_in)

#Scores outputted

scores = model(sent1_embedding, sent2_embedding)

#Rescaling scores back to [0,5]

print(scores.item()*5)