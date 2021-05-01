
#
# Much of this code has been taken from orginal implementation of punGen
#
# https://github.com/hhexiy/pungen
#@inproceedings{he2019pun,
#     title={Pun Generation with Surprise},
#     author={He He and Nanyun Peng and Percy Liang},
#     booktitle={North American Association for Computational Linguistics (NAACL)},
#     year={2019}
# }

from nltk.corpus import wordnet as wn

from fairseq.data import EditDataset
from fairseq import  utils, tokenizer
import numpy as np

import argparse
from fairseq.sequence_scorer import SequenceScorer
from fairseq import tasks, utils, data

import itertools


class LMScorer(object):
    def __init__(self, task, scorer):
        self.task = task
        self.scorer = scorer
        self.use_cuda = False

    @classmethod
    def load_model(cls, path, cpu=False):
        args = argparse.Namespace(data=os.path.dirname(path), path=path, cpu=cpu, task='language_modeling',
                output_dictionary_size=-1, self_target=False, future_target=False, past_target=False)
        use_cuda = torch.cuda.is_available() and not cpu
        task = tasks.setup_task(args)
        models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)
        d = task.target_dictionary
        scorer = SequenceScorer(models, d)
        return cls(task, scorer, use_cuda)

    def score_sents(self, sents, tokenize=str.split):
        """Return log p at each word
        """
        itr = self.make_batches(sents, self.task.target_dictionary, self.scorer.models[0].max_positions(), tokenize=tokenize)
        results = self.scorer.score_batched_itr(itr, cuda=self.use_cuda)
        scores = []
        for id_, src_tokens, __, hypos in results:
            pos_scores = hypos[0]['positional_scores'].data.cpu().numpy()
            scores.append((int(id_.data.cpu().numpy()), pos_scores))
        # sort by id
        scores = [s[1] for s in sorted(scores, key=lambda x: x[0])]
        return scores

    def make_batches(self, lines, src_dict, max_positions, tokenize=str.split):
        tokens = [
            tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False, tokenize=tokenize).long()
            for src_str in lines
        ]
        lengths = np.array([t.numel() for t in tokens])

        # Load dataset
        # MonolingualDataset[i] = source, future_target, past_target
        # all targets are effectively ignored during inference
        dataset = data.MonolingualDataset(
                dataset=[(s[:-1], s[1:], None) for s in tokens],
                sizes=lengths, src_vocab=src_dict, tgt_vocab=src_dict,
                add_eos_for_other_targets=False, shuffle=False)
        itr = self.task.get_batch_iterator(
            dataset=dataset,
            max_tokens=100,
            max_sentences=5,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)

        return itr

class TypeRecognizer(object):
    tags = {
            'noun': wn.NOUN,
            'verb': wn.VERB,
            'adj': wn.ADJ,
            'adv': wn.ADV,
            }

    person_words = set(['we', 'he', 'she', 'i', 'you', 'they', 'who', 'him'])

    def __init__(self, max_num_senses=2, threshold=0.2):
        self.max_num_senses = max_num_senses
        self.threshold = threshold
        self.person = wn.synsets('person')[0]

    def get_type(self, word, tag):
        if word in self.person_words:
            return [self.person]
        pos = self.tags.get(tag)
        s = wn.synsets(word, pos=pos)
        return s

    def is_types(self, word, types, tag):
        types1 = types
        types2 = self.get_type(word, tag)
        scores = []
        for t1 in types1[:self.max_num_senses]:
            for t2 in types2[:self.max_num_senses]:
                scores.append(t1.path_similarity(t2))
        if not scores or max(scores) < self.threshold:
            return False
        return True


def makeFairseqDataset(templates, deleted_words, src_dict, max_positions, task):

    # make list of sentences with <placeholder> 
    temps = [tokenizer.Tokenizer.tokenize(temp, src_dict, add_if_not_exist=False, tokenize=lambda x: x).long() for temp in templates]

    # make list of the 'target' words in that placeholder
    deleted = [tokenizer.Tokenizer.tokenize(word, src_dict, add_if_not_exist=False, tokenize=lambda x: x).long() for word in deleted_words]

    # make list of dictionaries
    inputs = [{'template': temp, 'deleted': dw} for temp, dw in zip(temps, deleted)]
    lengths = np.array([t['template'].numel() for t in inputs])

    # merge into a FairSeq dataset    
    dataset = EditDataset(inputs, lengths, src_dict, insert="deleted", combine="embedding")

    itr = task.get_batch_iterator(dataset=dataset, max_tokens=100, max_sentences=5, max_positions=max_positions).next_epoch_itr(shuffle=False)
    return itr


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


def sentence_iterator(file_):
    ''' Iterates through each line of a provided file
        Splits lines by spaces
        Converts words in to tuple, (Word, Lemma, POS)
    '''
    with open(file_, 'r') as fp:
        for i, line in enumerate(fp):
            line = line.strip().split()
            words = []
            for w in line:
                tags = w.split('|')
                words.append(tags)
            yield words


"""
SkipGram - generator for skipgram model
"""
import os
import pickle
from nltk.corpus import wordnet as wn

import torch
from torch import LongTensor as LT
from torch import FloatTensor as FT
from fairseq.data.dictionary import Dictionary

from pungen.utils import get_lemma, STOP_WORDS
#from pungen.model import Word2Vec, SGNS


class SkipGram(object):
    def __init__(self, model, vocab, use_cuda):
        self.model = model
        self.vocab = vocab
        if use_cuda:
            self.model.cuda()
        self.model.eval()

    @classmethod
    def load_model(cls, vocab_path, model_path, embedding_size=300, cpu=False):
        d = Dictionary.load(vocab_path)
        vocab_size = len(d)
        model = Word2Vec(vocab_size=vocab_size, embedding_size=embedding_size)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=1, weights=None)
        sgns.load_state_dict(torch.load(model_path))
        sgns.eval()
        use_cuda = torch.cuda.is_available() and not cpu
        return cls(sgns, d, use_cuda)

    def predict_neighbors(self, word, k=20, masked_words=None):
        # take lemma because skipgram is trained on lemmas
        lemma = get_lemma(word)
        word = lemma

        owords = range(len(self.vocab))

        # NOTE: 0 is <Lua heritage> in fairseq.data.dictionary
        masked_inds = [self.vocab.index(word), self.vocab.unk(), self.vocab.eos(), 0] + [self.vocab.index(w) for w in STOP_WORDS]
        if masked_words:
            masked_inds += [self.vocab.index(w) for w in masked_words]
        masked_inds = set(masked_inds)
        owords = [w for w in owords if not w in masked_inds and self.vocab.count[w] > 100]
        neighbors = self.topk_neighbors([word], owords, k=k)

        return neighbors

    def score(self, iwords, owords, lemma=False):
        """p(oword | iword)
        """
        if not lemma:
            iwords = [get_lemma(w) for w in iwords]
            owords = [get_lemma(w) for w in owords]
        iwords = [self.vocab.index(w) for w in iwords]
        owords = [self.vocab.index(w) for w in owords]
        ovectors = self.model.embedding.forward_o(owords)
        ivectors = self.model.embedding.forward_i(iwords)
        scores = torch.matmul(ovectors, ivectors.t())
        probs = scores.squeeze().sigmoid()
        return probs.data.cpu().numpy()

    # TODO: use self.score()
    def topk_neighbors(self, words, owords, k=10):
        """Find words in `owords` that are neighbors of `words` and are similar to `swords`.
        """
        vocab = self.vocab
        iwords = [vocab.index(word) for word in words]
        for iword, w in zip(iwords, words):
            if iword == vocab.unk():
                return []

        ovectors = self.model.embedding.forward_o(owords)
        scores = 0
        for iword in iwords:
            ivectors = self.model.embedding.forward_i([iword])
            score = torch.matmul(ovectors, ivectors.t())
            scores += score
        probs = scores.squeeze()#.sigmoid()

        topk_prob, topk_id = torch.topk(probs, min(k, len(owords)))
        return [vocab[owords[id_]] for id_ in topk_id]


"""
From model.py - Word2Vec and SGNS
"""
import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.nn.functional import logsigmoid as ls

class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None, pad=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        self.pad = pad
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        non_pad = FT((owords != self.pad).float())
        non_pad = non_pad.cuda() if self.embedding.ovectors.weight.is_cuda else non_pad
        N = non_pad.sum()
        nvectors = self.embedding.forward_o(nwords).neg()
        #oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        oloss = t.sum(ls(t.bmm(ovectors, ivectors).squeeze()) * non_pad) / N
        nloss = ls(t.bmm(nvectors, ivectors).squeeze()).view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)



