import torch
import torch.autograd as Variable
import torch.nn as nn
from torch.utils.data import dataloader
import torchvision
import torch.optim as optim
import torch.utils.data
import numpy as np
import pickle
import re
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

#Reading word embeddings

word_embeddings_file = open("Pickle/word_embeddings.pkl",'rb')
WORD_EMBEDDINGS = pickle.load(word_embeddings_file)
word_embeddings_file.close()

word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

#Reading vocabulary

vocabulary_file = open("Pickle/vocabulary_file.pkl",'rb')
VOCABULARY = pickle.load(vocabulary_file)
vocabulary_file.close()

#Dimension of Input Word Vectors
H_in = 300

#Dimension of Hidden Representations, Cell states (as used in the paper)
H_hidden = 50

#Batch_size
batch_size = 64

#Learning Rate
alpha = 0.001

#Number of epochs
num_epochs = 50

#Number of layers (Stacked vs Non stacked)
num_layers = 1

"""
We have 9840 training instances in our training set
"""

#Training Set Size
training = 8000

#Test Set Size
test = 1840

#Dataset Size
dataset = 9840

"""
Function to generate token
"""

def generateTokens(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


"""
Input will be (Batch_Size, Sequence Length, Input Dimension)
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

        

"""
Data Pre-processing
"""

file = open("SICK.txt", 'r')

sentences = []

lines = file.readlines()

"""
Preparing Training Set for the training loop
"""
#variable to compute length of longest sentence to fix sequence leng
leng_long = 0

for line in lines[1:]:

    #splitting by tab space
    line = line.split('\t')

    #tokenizing sentence 1
    sent1_tokenized = generateTokens(line[1])

    #Finding the longest sentence
    leng_long = len(sent1_tokenized) if len(sent1_tokenized) > leng_long else leng_long

    #storing word vectors of sentence 1
    sent1_embedding = []

    for word in sent1_tokenized:
        if word in word2vec_model:
            sent1_embedding.append(torch.from_numpy(word2vec_model[word]))
        else:
            sent1_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]])

    #tokenizing sentence 2
    sent2_tokenized = generateTokens(line[2])

    #Finding the longest sentence
    leng_long = len(sent2_tokenized) if len(sent2_tokenized) > leng_long else leng_long

    #storing word vectors of sentence 2
    sent2_embedding = []

    for word in sent2_tokenized:
        if word in word2vec_model:
            sent2_embedding.append(torch.from_numpy(word2vec_model[word]))
        else:
            sent2_embedding.append(WORD_EMBEDDINGS[VOCABULARY[word]]) 

    #storing the sentence embeddings in Training Set
    sentences.append((sent1_embedding, sent2_embedding, torch.tensor(float(line[4]))))

LONGEST_LENGTH = leng_long


#At this point, remember all word embeddings and labels are converted into Tensors!

"""
Pad with zeros so all sequence lengths are equal to LONGEST_LENGTH
"""
for idx in range(dataset):
    if len(sentences[idx][0]) < LONGEST_LENGTH:
        for _ in range(LONGEST_LENGTH - len(sentences[idx][0])):
            sentences[idx][0].append(torch.zeros(H_in))
    if len(sentences[idx][1]) < LONGEST_LENGTH:
        for _ in range(LONGEST_LENGTH - len(sentences[idx][1])):
            sentences[idx][1].append(torch.zeros(H_in))

"""
Shuffling and Splitting data into test and training data
"""

lst_indices = list(range(dataset))

random.shuffle(lst_indices)

train_data = [sentences[idx] for idx in lst_indices[:training]]

test_data = [sentences[idx] for idx in lst_indices[training:]]

"""
Loading data into batches for the training loop
"""

num_batches = math.ceil(training/batch_size)

train_loader = []

s_idx = 0
e_idx = batch_size


for _ in range(num_batches):
    train_loader.append(train_data[s_idx : e_idx])
    s_idx = e_idx
    e_idx = e_idx + batch_size if e_idx + batch_size <= training else e_idx + training % batch_size

#We now have our data segmented into training and test sets and training data loaded into mini-batches for the training loop as well

#All sequence lengths are also fixed to LONGEST_LENGTH

#Initializing Hidden & Cell States using a Multivariate Gaussian
hidden_state = torch.randn(num_layers, batch_size, H_hidden)

cell_state = torch.randn(num_layers, batch_size, H_hidden)

model = Siamese(H_in, H_hidden)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = alpha)

losses = []

print("All Pre-processing Done. Onto Training!!!")

for epoch in range(num_epochs):
    n_batches = 0
    running_loss = 0
    for idx, data in enumerate(tqdm(train_loader)):

        #Data being passed to the Siamese Network should be of the form (batch_size, LONGEST_LENGTH, H_in)

        list_sent1 = []

        list_sent2 = []

        label = torch.empty(len(data))

        for idx in range(len(data)):
            list_sent1.append(torch.stack(data[idx][0]))
            list_sent2.append(torch.stack(data[idx][1]))

            #Rescaling labels to [0,1]
            label[idx] = data[idx][2]/5
        
        sent1 = torch.stack(list_sent1)

        sent2 = torch.stack(list_sent2)

        scores = model.forward(sent1, sent2)

        loss = criterion(scores, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        n_batches += 1

    print("For epoch {}, Average Loss - {}".format(epoch + 1, running_loss/n_batches))

    losses.append(running_loss/n_batches)

#Making a plot of epochs vs losses
x = np.arange(1, num_epochs + 1)

plt.plot(x, losses)

plt.show()

#Saving trained Model
torch.save(model.state_dict(), "model_word2vec.pt")
print("Model Saved Successfully")