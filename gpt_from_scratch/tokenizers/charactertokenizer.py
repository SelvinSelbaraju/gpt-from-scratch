import torch
from gpt_from_scratch.tokenizers.basetokenizer import BaseTokenizer


# The vocab has one entry per character
# Here we have a small vocab but long sequneces for a given string
class CharacterTokenizer(BaseTokenizer):
    def encode(self, text: str) -> torch.Tensor:
        idx_list = [self.char_to_idx[char] for char in text]
        return torch.tensor(idx_list, dtype=torch.long)

    def decode(self, token_tensor: torch.Tensor) -> str:
        char_list = [self.idx_to_char[token] for token in token_tensor.numpy()]
        return "".join(char_list)
