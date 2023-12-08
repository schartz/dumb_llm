from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F

book_path = '/home/schartz/Documents/Books/wizard_of_oz.txt'

text = ''
with open(book_path, 'r', encoding='utf-8') as file:
    text = file.read()
# print(len(text))
# print(text[:200])


"""
 1. create a tokenizer
    tokenizer -> encoder and decoder
"""

# this is vocabulary
chars = sorted(set(text))
str_to_int = {c: i for i, c in enumerate(chars)}
int_to_str = {i: c for i, c in enumerate(chars)}
vocabulary_size = len(chars)


# encoder = lambda s: [str_to_int[c] for c in s]
def encoder(input_str: str) -> List[int]:
    return [str_to_int[character] for character in input_str]


# decoder = lambda l: ''.join([int_to_str[i] for i in l])
def decoder(int_list: List[int]) -> str:
    return ''.join([int_to_str[number] for number in int_list])


# print(encoder('Schartz Rehan'), ' <==> ', decoder([43, 56, 61, 54, 71, 73, 79, 1, 42, 58, 61, 54, 67]))


"""
 2. load our text data into an encoded tensor
    and create train test splits
"""

data = torch.tensor(encoder(text), dtype=torch.long)
# print(data[:150])
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]
# print(train_data[:20])
# print(val_data[:20])


# now lets see an example of what the next number (token) in our target data, 
# based on the numbers in our training data

BLOCK_SIZE = 8
x = train_data[:BLOCK_SIZE]
y = val_data[1: BLOCK_SIZE + 1]

for step in range(BLOCK_SIZE):
    context = x[:step + 1]
    target = y[step]
# print('with the input of: ', [context], ', prediction is: ', target)
# here we did everything one by one, on cpu. 
# We could run these block based predictions in parallel on GPU
# block size is the length of the sequence, 
# batch size is how many of the sequences are there at one time (typically in the GPU)
BATCH_SIZE = 4

# Let's check for the GPU here
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i))


"""
 3. A function to get batches randomly from the data
"""
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # print(ix)

    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# x, y = get_batch('train')
# print('inputs: ')
# print(x)
# print('targets: ')
# print(y)

"""
4.  Creating a simple Bigram model.
    This model predicts the next token based on the previous token in a given set of token sequence.
    This given token sequence is called "context"
"""
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        if targets is None:
            loss = None
        else:
            # B -> Batch
            # T -> time, it represents the order of a token(in this case a character) in the sequence of tokens
            # we call it time dimension. It runs from 0 to the length of token sequence.
            # C -> channels, this is a number which equals the vocabulary size
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # calculate predictions
            logits, loss = self.forward(index)

            # focus only on the last step
            logits = logits[:, -1, :]   # becomes (B, C)

            # apply softmax
            probablities = F.softmax(logits, dim=-1)    # (B, C)

            # take sample from the next distribution
            index_next = torch.multinomial(probablities, num_samples=1) # (B, 1)

            # append the sampled index into the running sequence
            index = torch.cat((index, index_next), dim=1)   # (B, T+1)
        return index


model = BigramLanguageModel(vocabulary_size)
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decoder(m.generate(context, max_new_tokens=500)[0].tolist())
print("printing result")
print()
print(generated_chars)
print("============================")
print()

"""
5.  Creating the optimizer for above Bigram model
"""
LEARNING_RATE = 3e-4
TRAIN_LOOP_COUNTS = 1000
EVAL_LOOP_COUNTS = 250

@torch.no_grad
def estimate_loss():
    l = {}
    # put the model in evaluation mode
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_LOOP_COUNTS)
        for k in range(EVAL_LOOP_COUNTS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        l[split] = losses.mean()
    # put the model back into training mode
    model.train()
    return l

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print("now training")
for iter in range(TRAIN_LOOP_COUNTS):
    if iter % EVAL_LOOP_COUNTS == 0:
        estimated_loss = estimate_loss()
        print(f"iteration: {iter}, train_loss: {estimated_loss['train']:.3f}, eval_loss: {estimated_loss['eval']:.3f}")


    # sample a batch of data
    xb, xy = get_batch("train")

    # evaluate loss
    logits, loss = model.forward(xb, xy)
    # this ensures that the gradients from previous runs do not accumulate by addition on each subsequent training runs
    # we reset the gradients on each iteration of this train loop.
    # generally we use set_to_none=False in case of RNNs
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(f"iteration: {iter}, train_loss: {estimated_loss['train']:.3f}, eval_loss: {estimated_loss['eval']:.3f}")



# This will still print garbage
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decoder(m.generate(context, max_new_tokens=500)[0].tolist())
print("printing result after training")
print()
print(generated_chars)
print("============================")
print()

