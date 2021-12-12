import numpy as np
import os
import torch

def encode_data(path, out_path):
    lines = []
    with open(path) as f:
        lines = f.readlines()
    l_split = [l.split() for l in lines]
    unique_chars = np.unique(np.array(l_split))
    vocab = {c:i for i,c in enumerate(unique_chars)}
    encoded = torch.tensor([[vocab[c] for c in l] for l in l_split]).type(torch.LongTensor)
    torch.save(encoded, out_path)
    
    vocab_size = len(vocab)
    max_seq_len = encoded.shape[1]
    start_letter = 0
    
    return vocab_size, max_seq_len, start_letter