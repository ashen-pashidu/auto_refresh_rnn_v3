#!/usr/bin/env python3
# benchmark_auto_refresh_rnn_v3.py
# Self-contained benchmark: AutoRefreshRNN_V3 vs LSTM vs Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Synthetic Dataset
# -------------------------------------------------
class LongSequenceDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=4096, num_samples=512):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.roll(self.data, shifts=-1, dims=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# -------------------------------------------------
# Baseline LSTM
# -------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.out(out)

# -------------------------------------------------
# Transformer Encoder
# -------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads=8, num_layers=2, ff_mult=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, 8192, hidden_size)/100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*ff_mult,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x) + self.pos_emb[:, :x.size(1), :]
        out = self.transformer(x)
        return self.out(out)

# -------------------------------------------------
# AutoRefreshRNN_V3 (Embedded)
# -------------------------------------------------
class AutoRefreshRNN_V3(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        chunk_size=256,
        memory_size=512,
        memory_compress_factor=4,
        mem_attn_heads=4,
        alpha_init=0.8,
        use_local_attention=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.memory_size = memory_size
        self.memory_compress_factor = max(1, memory_compress_factor)
        self.use_local_attention = use_local_attention

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size)

        self.W_xh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.W_hh.weight)

        self.refresh_gate_proj = nn.Linear(hidden_size*2, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.mem_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=mem_attn_heads, batch_first=True)
        if use_local_attention:
            self.local_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def _compress_memory(self, mem_tensor):
        b, mem_len, h = mem_tensor.shape
        if mem_len <= self.memory_size:
            return mem_tensor
        overflow = mem_len - self.memory_size
        group_len = self.memory_compress_factor
        n_groups = max(1, (overflow + group_len - 1)//group_len)
        to_compress = mem_tensor[:, : n_groups*group_len, :].reshape(b, n_groups, group_len, h).mean(dim=2)
        remaining = mem_tensor[:, n_groups*group_len:, :]
        new_mem = torch.cat([to_compress, remaining], dim=1)
        if new_mem.shape[1] > self.memory_size:
            new_mem = new_mem[:, -self.memory_size:, :]
        return new_mem

    def _process_chunk(self, embedded_chunk, h_prev):
        batch_size, chunk_len, hidden = embedded_chunk.shape
        h_expanded = h_prev.unsqueeze(1).expand(-1, chunk_len, -1)
        rec_parallel = self.W_hh(h_expanded)
        h_chunk = torch.tanh(self.W_xh(embedded_chunk)+rec_parallel) + embedded_chunk
        if self.use_local_attention:
            attn_out,_ = self.local_attn(h_chunk, h_chunk, h_chunk, need_weights=False)
            h_chunk = h_chunk + attn_out
        last = h_chunk[:, -1, :]
        mean_pool = h_chunk.mean(dim=1)
        chunk_summary = 0.5*(last+mean_pool)
        return h_chunk, chunk_summary

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device
        embedded = self.emb_norm(self.emb(x))
        num_chunks = (seq_len + self.chunk_size - 1)//self.chunk_size
        padded_len = num_chunks*self.chunk_size
        if padded_len > seq_len:
            padding = torch.zeros(batch_size, padded_len-seq_len, self.hidden_size, device=device)
            embedded = torch.cat([embedded, padding], dim=1)
        embedded_chunks = embedded.reshape(batch_size, num_chunks, self.chunk_size, self.hidden_size)

        memory = torch.zeros(batch_size, 0, self.hidden_size, device=device)
        h_prev = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []

        for idx in range(num_chunks):
            chunk = embedded_chunks[:, idx]
            h_chunk, chunk_summary = self._process_chunk(chunk, h_prev)
            if memory.shape[1] > 0:
                q = chunk_summary.unsqueeze(1)
                mem_out,_ = self.mem_attn(q, memory, memory, need_weights=False)
                h_prev = h_prev + mem_out.squeeze(1)
            gate_in = torch.cat([h_prev, chunk_summary], dim=-1)
            gate = torch.sigmoid(self.refresh_gate_proj(gate_in))
            chunk_last = h_chunk[:, -1, :]
            detached_last = chunk_last.detach()
            candidate = self.alpha*detached_last + (1.0-self.alpha)*chunk.mean(dim=1)
            h_prev = gate*candidate + (1.0-gate)*chunk_last
            memory = torch.cat([memory, chunk_summary.unsqueeze(1)], dim=1)
            if memory.shape[1] > self.memory_size:
                memory = self._compress_memory(memory)
            outputs.append(self.out(h_chunk))

        logits = torch.cat(outputs, dim=1)[:, :seq_len, :]
        return logits

# -------------------------------------------------
# Benchmark Function
# -------------------------------------------------
def benchmark(model, loader, name="Model", device="cuda"):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    total_loss, total_time, grad_norms = 0, 0, []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        start = time.time()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        grad_norms.append(grad_norm)
        optimizer.step()
        total_time += time.time()-start
        total_loss += loss.item()

    avg_loss = total_loss/len(loader)
    avg_grad = np.mean(grad_norms)
    avg_time = total_time/len(loader)
    print(f"\n{name} Benchmark:")
    print(f"  Avg Loss: {avg_loss:.4f} | Avg Grad: {avg_grad:.4f} | Time/Epoch: {avg_time:.2f}s")
    return avg_loss, avg_grad, avg_time

# -------------------------------------------------
# Main Benchmark
# -------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 512
    hidden_size = 256
    seq_len = 2048
    batch_size = 8
    dataset = LongSequenceDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    models = [
        (LSTMModel(vocab_size, hidden_size), "LSTM"),
        (TransformerModel(vocab_size, hidden_size, num_heads=4, num_layers=2), "TransformerEncoder"),
        (AutoRefreshRNN_V3(vocab_size, hidden_size, chunk_size=256, memory_size=256), "AutoRefreshRNN_V3"),
    ]

    results = []
    print(f"Running benchmarks on {device}\n{'-'*60}")
    for model, name in models:
        res = benchmark(model, loader, name, device)
        results.append((name, *res))

    print("\nSummary Table")
    print("-"*60)
    print(f"{'Model':25s} | {'Loss':>8s} | {'Grad':>8s} | {'Time(s)':>8s}")
    print("-"*60)
    for name, loss, grad, t in results:
        print(f"{name:25s} | {loss:8.4f} | {grad:8.4f} | {t:8.2f}")
    print("-"*60)

if __name__ == "__main__":
    main()
