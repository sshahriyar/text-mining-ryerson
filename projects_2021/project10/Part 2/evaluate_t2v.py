'''
For evaluating precision and recall metrics
'''
import numpy as np
import sys
import pickle as pkl
import io

import matplotlib.pyplot as plt

K1 = 1
K2 = 10

HIST = False

def precision(p, t, k):
    '''
    Compute precision @ k for predictions p and targets t
    '''

    n = p.shape[0]
    res = np.zeros(n)
    for idx,items in enumerate(t):
        index = p[idx,:]
        for i in items:
            if i in index:
                res[idx] += 1
    return np.sum(res)/(n*k)

def recall(p, t, k):
    '''
    Compute recall @ k for predictions p and targets k
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx,items in enumerate(t):
        index = p[idx,:]
        for i in items:
            if i in index:
                res[idx] += 1
        res[idx] = res[idx] / len(index)
    return np.sum(res)/n*k

def meanrank(p, t):
    '''
    Compute mean rank of targets in the predictions
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx, items in enumerate(t):
        ind = p[idx,:]
        minrank = p.shape[1]+1
        for i in items:
            currrank = np.where(ind==i)[0]+1
            if currrank < minrank:
                minrank = currrank
        res[idx] = minrank
    return int(np.mean(res)), res

def readable_predictions(p, t, d, k, labeldict):
    out = []
    for idx, item in enumerate(d):
        preds = p[idx,:]
        plabels = ','.join([list(labeldict.keys())[ii-1] if ii > 0 else '<unk>' for ii in preds])
        tlabels = ','.join([list(labeldict.keys())[ii-1] if ii > 0 else '<unk>' for ii in t[idx]])
        out.append('%s\t%s\t%s\n'%(tlabels,plabels,item))
    return out

def main(result_path):
    with open('%s/predictions-nlp-t2v-p2.npy'%result_path,'rb') as f:
        p = np.load(f)
    with open('%s/targets-nlp-t2v-p2.pkl'%result_path,'rb') as f:
        t = pkl.load(f)
    with open('%s/data-nlp-t2v-p2.pkl'%result_path,'rb') as f:
        d = pkl.load(f)
    with open('%s/embeddings-nlp-t2v-p2.npy'%result_path,'rb') as f:
        e = np.load(f)
    with open('%s/label_dict-nlp-t2v-p2.pkl'%result_path,'rb') as f:
        labeldict = pkl.load(f)

    readable = readable_predictions(p, t, d, 10, labeldict)
    with io.open('%s/readable-t2v.txt'%result_path,'w') as f:
        for line in readable:
            f.write(line)

    meanr, allr = meanrank(p,t)
    print("Precision @ {} = {}".format(K1,precision(p,t,K1)))
    print("Recall @ {} = {}".format(K2,recall(p,t,K2)))
    print("Mean rank = {}".format(meanr))

    # histogram
    if HIST:
        hist, bins = np.histogram(allr, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1])