from abc import ABC, abstractmethod
import torch


# Create a tokenizer to map characters to integers
# Many other tokenizers exist, which may map sub-words to integers
# There is a tradeoff between sequence length and vocab size
class BaseTokenizer(ABC):
    def __init__(self, vocab: list) -> None:
        self.vocab_length = len(vocab)
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, token_tensor: torch.Tensor) -> str:
        pass
