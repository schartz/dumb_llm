import torch
import torch.nn as nn
import torch.nn.functional as F

# initial variables
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
with open('/home/schartz/Documents/Books/wizard_of_oz.txt', 'r', encoding='utf=8') as f:
    text = f.read()
chars = sorted(set(text))

# hyper parameters
BLOCK_SIZE = 8
BATCH_SIZE = 4
MAX_ITERS = 1000
LR = 3e-3
EVAL_ITERS = 250
VOCAB_SIZE = len(chars)
NUM_EMBEDDINGS = 128    # dimensions of vector for each token
NUM_ND_CODERS = 4       # number of encoders/decoders in the network
NUM_ATTENTION_HEADS = 4

# variables for model
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_int[c] for c in s]
decode = lambda nums: ''.join([int_to_str[n] for n in nums])
data = torch.tensor(encode(text), dtype=torch.long)

# tarin test split and get_batch function
n = int(0.8*len(data))
train_data = data[:n]
eval_data = data[n:]
def get_batch(split: str):
    data = train_data if split == 'train' else eval_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# function for loss reporting calcutaions
@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# model definitaion for a simple GPT/Transformer model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBEDDINGS)

        # positional encodings refer to paper 
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, NUM_EMBEDDINGS)
        self.blocks = nn.Sequential(*[Block(NUM_EMBEDDINGS, NUM_ATTENTION_HEADS) for _ in range(NUM_ND_CODERS)])
        self.layer_norm_final = nn.LayerNorm(NUM_EMBEDDINGS)
        self.lm_head = nn.Linear(NUM_EMBEDDINGS, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            print("nothing to normalize in module", module)


    def forward(self, index, targets = None):
        logits: torch.Tensor = self.token_embedding_table(index)

        # index and targets are both tensors of integers with shape (B, T)
        token_embedding = self.token_embedding_table(index)     # (B, T, C)
        position_embedding = self.positional_embedding_table(torch.arange(T, device=device))   # (T, C)
        x = token_embedding + position_embedding    # (B, T, C), also refer to https://pytorch.org/docs/stable/notes/broadcasting.html
        x = self.blocks(x)      # (B, T, C)
        x = self.layer_norm_final(x)    # (B, T, C)
        logits = self.lm_head(x)    # (B, T, VOCAB_SIZE)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_index), dim=1)
        return index

    
model = GPTLanguageModel(VOCAB_SIZE)
model.to(device)


# train loop with optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
for iter in range(MAX_ITERS):
    if iter % EVAL_ITERS == 0:
        losses = estimate_loss(model)
        print(f"Step: {iter}, train_loss: {losses['train']:.3f}, eval_loss: {losses['eval']:.3f}")

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("final loss:")


## This will still print garbage
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("printing result after training")
print()
print(generated_chars)
print("============================")
print()


