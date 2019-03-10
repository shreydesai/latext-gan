import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from models import Autoencoder
from dataset import load

def train(epoch):
    model.train()
    train_loss = 0.
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        if args.cuda:
            x = x.cuda()
        _, logits = model(x)
        loss = criterion(logits, x)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if args.interval > 0 and i % args.interval == 0:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(train_loader.dataset),
                100.*(args.batch_size*i)/len(train_loader.dataset),
                loss.item()
            ))
    train_loss /= len(train_loader)
    print('* (Train) Epoch: {} | Loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seq-len', type=int, default=20+2)
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--enc-hidden-dim', type=int, default=100)
    parser.add_argument('--dec-hidden-dim', type=int, default=600)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, vocab = load(args.batch_size, args.seq_len)

    model = Autoencoder(args.enc_hidden_dim, args.dec_hidden_dim, args.embedding_dim,
                        args.latent_dim, vocab.size(), args.dropout, args.seq_len)
    
    if args.cuda:
        model = model.cuda()
    
    print('Parameters:', sum([p.numel() for p in model.parameters() if \
                              p.requires_grad]))
    print('Vocab size:', vocab.size())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        if loss < best_loss:
            best_loss = loss
            print('* Saved')
            torch.save(model.state_dict(), 'autoencoder.th')