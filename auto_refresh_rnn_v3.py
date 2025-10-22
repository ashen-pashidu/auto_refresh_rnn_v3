#!/usr/bin/env python3
# auto_refresh_rnn_v3.py
# Advanced Unbounded Memory Auto-Refresh Recurrent Neural Network
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Input projection
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size)

        # Recurrent update (orthogonal + residual)
        self.W_xh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.W_hh.weight)

        # Refresh gate
        self.refresh_gate_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # Cross-memory attention
        self.mem_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=mem_attn_heads,
            batch_first=True
        )

        # Optional local attention (for better token-level adaptation)
        if use_local_attention:
            self.local_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=2,
                batch_first=True
            )

        # Output projection
        self.out = nn.Linear(hidden_size, vocab_size)

    # -------------------------------------------------
    # Memory Compression Mechanism
    # -------------------------------------------------
    def _compress_memory(self, mem_tensor):
        """Compress memory when it exceeds memory_size."""
        b, mem_len, h = mem_tensor.shape
        if mem_len <= self.memory_size:
            return mem_tensor

        overflow = mem_len - self.memory_size
        group_len = self.memory_compress_factor
        n_groups = max(1, (overflow + group_len - 1) // group_len)

        # Mean pooling to compress oldest memory
        to_compress = mem_tensor[:, :n_groups * group_len, :].reshape(
            b, n_groups, group_len, h
        ).mean(dim=2)
        remaining = mem_tensor[:, n_groups * group_len :, :]

        new_mem = torch.cat([to_compress, remaining], dim=1)
        if new_mem.shape[1] > self.memory_size:
            new_mem = new_mem[:, -self.memory_size :, :]
        return new_mem

    # -------------------------------------------------
    # Process a chunk in parallel
    # -------------------------------------------------
    def _process_chunk(self, embedded_chunk, h_prev):
        b, chunk_len, h = embedded_chunk.shape

        h_expanded = h_prev.unsqueeze(1).expand(-1, chunk_len, -1)
        rec_parallel = self.W_hh(h_expanded)

        # Recurrent transformation + residual input
        h_chunk = torch.tanh(self.W_xh(embedded_chunk) + rec_parallel) + embedded_chunk

        # Optional local attention refinement
        if self.use_local_attention:
            attn_out, _ = self.local_attn(h_chunk, h_chunk, h_chunk, need_weights=False)
            h_chunk = h_chunk + attn_out

        # Summary representation (average + last token)
        last = h_chunk[:, -1, :]
        mean_pool = h_chunk.mean(dim=1)
        chunk_summary = 0.5 * (last + mean_pool)

        return h_chunk, chunk_summary

    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------
    def forward(self, x):
        b, seq_len = x.size()
        device = x.device

        embedded = self.emb_norm(self.emb(x))
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size

        if padded_len > seq_len:
            padding = torch.zeros(
                b, padded_len - seq_len, self.hidden_size, device=device
            )
            embedded = torch.cat([embedded, padding], dim=1)

        embedded_chunks = embedded.view(b, num_chunks, self.chunk_size, self.hidden_size)

        # Initialize recurrent states
        memory = torch.zeros(b, 0, self.hidden_size, device=device)
        h_prev = torch.zeros(b, self.hidden_size, device=device)
        outputs = []

        # Process sequence chunk-by-chunk
        for idx in range(num_chunks):
            chunk = embedded_chunks[:, idx]
            h_chunk, chunk_summary = self._process_chunk(chunk, h_prev)

            # Memory attention update
            if memory.shape[1] > 0:
                q = chunk_summary.unsqueeze(1)
                mem_out, _ = self.mem_attn(q, memory, memory, need_weights=False)
                h_prev = h_prev + mem_out.squeeze(1)

            # Refresh gate mechanism
            gate_in = torch.cat([h_prev, chunk_summary], dim=-1)
            gate = torch.sigmoid(self.refresh_gate_proj(gate_in))

            # Candidate state with alpha blending
            chunk_last = h_chunk[:, -1, :]
            detached_last = chunk_last.detach()
            candidate = self.alpha * detached_last + (1.0 - self.alpha) * chunk.mean(dim=1)

            # Smoothly blend new and old states
            h_prev = gate * candidate + (1.0 - gate) * chunk_last

            # Update memory
            memory = torch.cat([memory, chunk_summary.unsqueeze(1)], dim=1)
            if memory.shape[1] > self.memory_size:
                memory = self._compress_memory(memory)

            # Output logits for current chunk
            outputs.append(self.out(h_chunk))

        logits = torch.cat(outputs, dim=1)[:, :seq_len, :]
        return logits
