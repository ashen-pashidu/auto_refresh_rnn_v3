#!/usr/bin/env python3
# inference.py — Demo inference for AutoRefreshRNN_V3

import torch
from auto_refresh_rnn_v3 import AutoRefreshRNN_V3

@torch.no_grad()
def run_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRefreshRNN_V3(vocab_size=512, hidden_size=256, chunk_size=256).to(device)
    model.eval()

    # Random input (simulate tokenized sequence)
    seq = torch.randint(0, 512, (1, 1024), device=device)
    logits = model(seq)

    next_token = torch.argmax(logits[0, -1]).item()
    print(f"Predicted next token: {next_token}")

if __name__ == "__main__":
    run_demo()
