import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from models import Autoencoder, Generator, Critic
from dataset import load

def compute_grad_penalty(critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    if args.cuda:
        alpha = alpha.cuda()
    sample = alpha*real_data + (1-alpha)*fake_data
    sample.requires_grad_(True)
    score = critic(sample)

    outputs = torch.FloatTensor(B, args.latent_dim).fill_(1.0)
    outputs.requires_grad_(False)
    if args.cuda:
        outputs = outputs.cuda()
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    #grads = grads.view(B, -1)
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty

def train(epoch):
    autoencoder.eval()
    generator.train()
    critic.train()
    c_train_loss = 0.
    g_train_loss = 0.
    g_batches = 0
    for i, x in enumerate(train_loader):
        if args.cuda:
            x = x.cuda()
        
        # train critic
        B = x.size(0)
        c_optimizer.zero_grad()
        noise = torch.from_numpy(np.random.normal(0, 1, (B,
                                 args.latent_dim))).float()
        if args.cuda:
            noise = noise.cuda()
        with torch.no_grad():
            z_real = autoencoder(x)[0]
        z_fake = generator(noise)
        real_score = critic(z_real)
        fake_score = critic(z_fake)
        grad_penalty = compute_grad_penalty(critic, z_real.data, z_fake.data)
        c_loss = -torch.mean(real_score) + torch.mean(fake_score) + \
                 args.gp_lambda*grad_penalty
        c_train_loss += c_loss.item()
        c_loss.backward()
        c_optimizer.step()

        # train generator
        if i % args.n_critic == 0:
            g_batches += 1
            g_optimizer.zero_grad()
            fake_score = critic(generator(noise))
            g_loss = -torch.mean(fake_score)
            g_train_loss += g_loss.item()
            g_loss.backward()
            g_optimizer.step()
        
        if args.interval > 0 and i % args.interval == 0:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | G Loss: {:.6f} | C Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(train_loader.dataset),
                100.*(args.batch_size*i)/len(train_loader.dataset),
                g_loss.item(), c_loss.item()
            ))
    g_train_loss /= g_batches
    c_train_loss /= len(train_loader)
    print('* (Train) Epoch: {} | G Loss: {:.4f} | C Loss: {:.4f}'.format(
        epoch, g_train_loss, c_train_loss
    ))
    return (g_train_loss, c_train_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seq-len', type=int, default=20+2)
    parser.add_argument('--gp-lambda', type=int, default=10)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--block-dim', type=int, default=100)
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--enc-hidden-dim', type=int, default=100)
    parser.add_argument('--dec-hidden-dim', type=int, default=600)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, vocab = load(args.batch_size, args.seq_len)

    autoencoder = Autoencoder(args.enc_hidden_dim, args.dec_hidden_dim, args.embedding_dim,
                              args.latent_dim, vocab.size(), args.dropout, args.seq_len)
    autoencoder.load_state_dict(torch.load('autoencoder.th', map_location=lambda x,y: x))
    generator = Generator(args.n_layers, args.block_dim)
    critic = Critic(args.n_layers, args.block_dim)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    c_optimizer = optim.Adam(critic.parameters(), lr=args.lr)
    
    if args.cuda:
        autoencoder = autoencoder.cuda()
        generator = generator.cuda()
        critic = critic.cuda()
    
    print('G Parameters:', sum([p.numel() for p in generator.parameters() if \
                                p.requires_grad]))
    print('C Parameters:', sum([p.numel() for p in critic.parameters() if \
                                p.requires_grad]))
    
    best_loss = np.inf
    
    for epoch in range(1, args.epochs + 1):
        g_loss, c_loss = train(epoch)
        loss = g_loss + c_loss
        if loss < best_loss:
            best_loss = loss
            print('* Saved')
            torch.save(generator.state_dict(), 'generator.th')
            torch.save(critic.state_dict(), 'critic.th')