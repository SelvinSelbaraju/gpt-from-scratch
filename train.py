import torch

from gpt_from_scratch.config import v1
from gpt_from_scratch.tokenizers.charactertokenizer import CharacterTokenizer
from gpt_from_scratch.tokenizers.utils import create_char_vocab
from gpt_from_scratch.trainers.bigramtrainer import BigramTrainer

with open("data/input.txt") as f:
    text = f.read()

# Create a list of unique characters
vocab = create_char_vocab(text)

tokenizer = CharacterTokenizer(vocab)

# Check encoder/decoder
# test_str = "Hi there!"
# encoded = tokenizer.encode(test_str)
# decoded = tokenizer.decode(encoded)
# print(f"Encoded: {encoded}")
# print(f"Decoded: {decoded}, decoded matches original: {test_str == decoded}")

# Tokenize the entire text into a tensor
# Split into train/test
data = tokenizer.encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Feed data in random chunks
# We go through the chunk. Given each of the previous tokens
# We predict the next one. Therefore, each block actually needs 9 tokens
# Transformer learns on contexts up to block size
# This is useful when string is less than block size
# Beyond block size, the transformer truncates the inference to the end
# block_size = 8
# x = train[:block_size]
# y = train[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     # Just t because y starts from 1
#     # If t = 0, x=train[0], y=train[1]
#     target = y[t]
#     print(f"When context is {context}, target is {target}")

torch.manual_seed(1337)

trainer = BigramTrainer(vocab, v1.model_params)
trainer.train(train_data, val_data)

example_idx = torch.zeros((1, 1), dtype=torch.long)
print(
    tokenizer.decode(
        trainer.model.generate(example_idx, max_new_tokens=500)[0]
    )
)
