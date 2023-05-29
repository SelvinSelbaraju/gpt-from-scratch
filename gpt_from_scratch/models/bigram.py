import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, block_size: int, vocab_size: int) -> None:
        # Call the init method on nn.Module
        super().__init__()
        self.block_size = block_size

        # For each token, we have the prob of all other tokens being next
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Returns (B, T, embedding_dim) for each token
        logits = self.token_embedding_table(idx)

        # Forward pass for training or inference
        # Cant pass targets for inference
        # So loss is None, and logits aren't reshaped
        if targets is None:
            loss = None
        else:
            # Calculate cross entropy loss for these logits
            # Each row of logits represents a score for each class
            # Targets are the class indices
            # loss expects logits as (B, num_classes, T)
            B, T, C = logits.shape
            # Each token is an independent batch
            # Look at the logits score for each batch
            logits = logits.view(B * T, C)
            # Need to put targets as independent batches too
            # For each batch (single token), we have the true next token
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Generate an additional max_new_tokens for a block
    def generate(self, idx, max_new_tokens):
        # Add a new character on the end multiple times
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on last time step for adding new token
            # This will have one per random block
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from distribution for each batch
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to end of sample block
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
