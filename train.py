#!/usr/bin/env python3
# train.py — Example training script for AutoRefreshRNN_V3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from auto_refresh_rnn_v3 import AutoRefreshRNN_V3


# Synthetic dataset
class LongSequenceDataset(Dataset):
    def __init__(self, vocab_size=512, seq_len=2048, num_samples=256):
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.roll(self.data, shifts=-1, dims=1)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRefreshRNN_V3(vocab_size=512, hidden_size=256, chunk_size=256, memory_size=256).to(device)
    loader = DataLoader(LongSequenceDataset(), batch_size=8, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    train()
