import numpy as np
import pickle as pkl
import codecs

from collections import OrderedDict
from settings_word import MAX_LENGTH, WORD_LEVEL

class BatchTweets():

    def __init__(self, data, targets, labeldict, batch_size=128, max_classes=1000, test=False):
        # convert targets to indices
        if not test:
            tags = []
            for l in targets:
                tags.append(labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0)
        else:
            tags = []
            for line in targets:
                tags.append([labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0 for l in line])

        self.batch_size = batch_size
        self.data = data
        self.targets = tags

        self.prepare()
        self.reset()

    def prepare(self):
        self.indices = np.arange(len(self.data))
        self.curr_indices = np.random.permutation(self.indices)

    def reset(self):
        self.curr_indices = np.random.permutation(self.indices)
        self.curr_pos = 0
        self.curr_remaining = len(self.curr_indices)

    def __next__(self):
        if self.curr_pos >= len(self.indices):
            self.reset()
            raise StopIteration()

        # current batch size
        curr_batch_size = np.minimum(self.batch_size, self.curr_remaining)

        # indices for current batch
        curr_indices = self.curr_indices[self.curr_pos:self.curr_pos+curr_batch_size]
        self.curr_pos += curr_batch_size
        self.curr_remaining -= curr_batch_size

        # data and targets for current batch
        x = [self.data[ii] for ii in curr_indices]
        y = [self.targets[ii] for ii in curr_indices]
        
#         print("inside class Batch Tweets x:", x)
#         print("inside class Batch Tweets y:", y)

        return x, y

    def __iter__(self):
        return self

def prepare_data(seqs_x, tokendict, n_tokens=1000):
    """
    Prepare the data for training - add masks and remove infrequent tokens
    """
    seqsX = []
    for cc in seqs_x:
        #print("inside batch prepare data cc:",cc)
        if (WORD_LEVEL):
            #print("seqsX:", seqsX)
            seqsX.append([tokendict[c] if c in tokendict and tokendict[c] < n_tokens else 0 for c in cc.split()[:MAX_LENGTH]])
        else:
            seqsX.append([tokendict[c] if c in tokendict and tokendict[c] < n_tokens else 0 for c in list(cc)[:MAX_LENGTH]])
    seqs_x = seqsX

    lengths_x = [len(s) for s in seqs_x]
    #print("lengths_x:",lengths_x)

    n_samples = len(seqs_x)
    #print("n_samples:", n_samples)

    x = np.zeros((n_samples,MAX_LENGTH)).astype('int32')
    #print("inside batch prepare x:" ,x)
    x_mask = np.zeros((n_samples,MAX_LENGTH)).astype('float32')
    #print("inside batch prepare x_mask:" ,x_mask)
    for idx, s_x in enumerate(seqs_x):
#         print("idx:",idx)
#         print("s_x:",s_x)
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.
#     print("Result x:" , x)
#     print("Expand x:", np.expand_dims(x, axis=2))
#     print("Result x_mask:" , x_mask)

    return np.expand_dims(x, axis=2), x_mask

def build_dictionary(text):
    """
    Build a dictionary of characters or words
    text: list of tweets
    """
    tokencount = OrderedDict()
    #print("inside batch, tokencount:", tokencount)

    for cc in text:       
        if WORD_LEVEL:
            tokens = cc.split()
        else:
            tokens = list(cc)
        for c in tokens:
            if c not in tokencount:
                tokencount[c] = 0
            tokencount[c] += 1

    tokens = list(tokencount.keys())
    #print("inside batch, tokens:", tokens)
    freqs = list(tokencount.values())
    #print("inside batch, freqs:", freqs)
    sorted_idx = np.argsort(freqs)[::-1]
    #print("inside batch sorted_idx",sorted_idx)

    tokendict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        #print(idx, sidx)
        tokendict[tokens[sidx]] = idx + 1

    return tokendict, tokencount

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

def build_label_dictionary(targets):
    """
    Build a label dictionary
    targets: list of labels, each item may have multiple labels
    """
    labelcount = OrderedDict()
    for l in targets:
        if l not in labelcount:
            labelcount[l] = 0
        labelcount[l] += 1
    labels = list(labelcount.keys())
    freqs = list(labelcount.values())
    sorted_idx = np.argsort(freqs)[::-1]

    labeldict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        labeldict[labels[sidx]] = idx + 1

    return labeldict, labelcount