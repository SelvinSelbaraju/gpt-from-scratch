import torch

from gpt_from_scratch.config import v1
from gpt_from_scratch.data.utils import get_batch
from gpt_from_scratch.tokenizers.charactertokenizer import CharacterTokenizer
from gpt_from_scratch.tokenizers.utils import create_char_vocab
from gpt_from_scratch.trainers.bigramtrainer import BigramTrainer

with open("data/input.txt") as f:
    text = f.read()

# Create a list of unique characters
vocab = create_char_vocab(text)

tokenizer = CharacterTokenizer(vocab)

# Check encoder/decoder
# test_str = "Hi there!"
# encoded = tokenizer.encode(test_str)
# decoded = tokenizer.decode(encoded)
# print(f"Encoded: {encoded}")
# print(f"Decoded: {decoded}, decoded matches original: {test_str == decoded}")

# Tokenize the entire text into a tensor
# Split into train/test
data = tokenizer.encode(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Feed data in random chunks
# We go through the chunk. Given each of the previous tokens
# We predict the next one. Therefore, each block actually needs 9 tokens
# Transformer learns on contexts up to block size, useful for inference of various size
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
trainer.train(train_data)

# Throw in 100 random batches of 32 and get average loss
cnt_val_batches = 100
val_losses = torch.zeros((cnt_val_batches,))
for i in range(cnt_val_batches):
    xb_val, yb_val = get_batch(val_data, 1, v1.model_params["block_size"])

    _, val_loss = trainer.model(xb_val, yb_val)
    val_losses[i] = val_loss

print(f"Validation losses: {torch.mean(val_losses)}")
example_idx = torch.zeros((1,1), dtype=torch.long)
print(tokenizer.decode(trainer.model.generate(example_idx, max_new_tokens=500)[0]))






