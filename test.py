import torch
import torch.nn as nn
import numpy as np

from models import Autoencoder, Generator
from dataset import Corpus

ds = Corpus()
vocab = ds.vocab

generator = Generator(20, 100)
generator.eval()
generator.load_state_dict(torch.load('generator.th', map_location='cpu'))

autoencoder = Autoencoder(100, 600, 200, 100, vocab.size(), 0.5, 22)
autoencoder.eval()
autoencoder.load_state_dict(torch.load('autoencoder.th', map_location='cpu'))

# sample noise
noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
z = generator(noise[None,:,:])

# create new sent
logits = autoencoder.decode(z).squeeze()
seq = logits.argmax(dim=0)
print(ds.decode(seq))