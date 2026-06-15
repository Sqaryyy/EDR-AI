import torch
import torch.nn as nn
import numpy as np
from sequences import log_to_sequence, sequence_to_ngrams, SYSCALLS, VOCAB_SIZE
from train import SyscallLSTM, N, EMBED_DIM, HIDDEN_DIM
import sys

# load model
model = SyscallLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("model/lstm.pt"))
model.eval()

def score_sequence(filepath):
    seq = log_to_sequence(filepath)
    ngrams = sequence_to_ngrams(seq, n=N)
    
    if len(ngrams) == 0:
        print(f"[WARN] {filepath} too short to score")
        return

    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for x, y in ngrams:
            x_tensor = torch.tensor([x], dtype=torch.long)
            y_tensor = torch.tensor([y], dtype=torch.long)
            out = model(x_tensor)
            loss = criterion(out, y_tensor)
            losses.append(loss.item())

    avg_loss = np.mean(losses)
    max_loss = np.max(losses)
    
    # threshold tuned from normal logs
    THRESHOLD = 0.55
    status = "ANOMALY" if avg_loss > THRESHOLD else "NORMAL"
    
    print(f"[{status}] {filepath}")
    print(f"  avg prediction loss: {avg_loss:.4f}")
    print(f"  max prediction loss: {max_loss:.4f}")
    print(f"  syscalls analyzed:   {len(ngrams)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 detect.py <logfile>")
        sys.exit(1)
    score_sequence(sys.argv[1])