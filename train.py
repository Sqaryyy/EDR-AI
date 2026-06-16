import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sequences import build_and_save_vocab, load_normal_logs
import os

N = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001

class SyscallLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    print("Building vocab...")
    syscall_to_idx, syscalls = build_and_save_vocab()
    vocab_size = len(syscalls)
    print(f"Vocab size: {vocab_size}")

    print("Loading sequences...")
    X, Y = load_normal_logs("logs", syscall_to_idx, n=N)

    X_tensor = torch.tensor(X, dtype=torch.long)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SyscallLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training LSTM...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/lstm.pt")
    print("Model saved to model/lstm.pt")