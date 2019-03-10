import collections

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Vocab:

    def __init__(self, corpus):
        self.words = self._build(corpus)
        self.encoding = {w:i for i,w in enumerate(self.words, 1)}
        self.decoding = {i:w for i,w in enumerate(self.words, 1)}

        self.register('<pad>', 0)
        self.register('<unk>')
        self.register('<s>')
        self.register('</s>')
    
    def _build(self, corpus, clip=1):
        vocab = collections.Counter()
        for sent in corpus:
            vocab.update(sent)
        
        for word in list(vocab.keys()):
            if vocab[word] < clip:
                vocab.pop(word)
        
        return list(sorted(vocab.keys()))

    def register(self, token, index=-1):
        i = len(self.encoding) if index<0 else index
        self.encoding[token] = i
        self.decoding[i] = token

    def size(self):
        assert len(self.encoding) == len(self.decoding)
        return len(self.encoding)

class Corpus(Dataset):
    
    def __init__(self, seq_len=20+2):
        self.seq_len = seq_len
        self.reviews, _ = self._load()
        self.vocab = Vocab(self.reviews)
    
    def _load(self):
        with open('sst.txt','r') as f:
            sents = [x for x in f.read().split('\n') if \
                     len(x.split())-1<=self.seq_len-2]
            reviews = [x.split()[1:] for x in sents]
            labels = [int(x.split()[0]) for x in sents]
        return (reviews, labels)
    
    def pad(self, sample):
        l,r = 0,self.seq_len-len(sample)
        return np.pad(sample, (0,r), 'constant')
    
    def encode(self, sample):
        enc = self.vocab.encoding
        unk_idx = enc['<unk>']
        return np.array([enc['<s>']]+[enc.get(c, unk_idx) \
                         for c in sample]+[enc['</s>']])
    
    def decode(self, sample):
        dec = self.vocab.decoding
        return ' '.join(np.array([dec[c.item()] for c in sample]))
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.pad(self.encode(self.reviews[i])))

def load(batch_size, seq_len):
    ds = Corpus(seq_len)
    return (DataLoader(ds, batch_size, shuffle=True), ds.vocab)