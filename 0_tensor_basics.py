import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
randint = torch.randint(-100, 100, (6,))
# print(randint)

tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
# print(tensor)

logspace = torch.logspace(start=-10, end=10, steps=5)
# print(logspace)

def timit(fn: callable):
	def inner():
		st = time.time()
		fn()
		et = time.time()
		et = et - st
		# print(f"execution time: {et}")
	return inner


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@timit
def crunch_numbers_cuda():
	print("matmul of 2 100, 100, 100, 100 random matrices on GPU")
	torch_rand1 = torch.rand(100, 100, 100, 100).to(device)
	torch_rand2 = torch.rand(100, 100, 100, 100).to(device)
	rand = (torch_rand1 @ torch_rand2)
	print("matmul completed on GPU")


@timit
def crunch_numbers_cpu():
	print("matmul of 2 100, 100, 100, 100 random matrices on CPU")
	np_rand1 = torch.rand(100, 100, 100, 100)
	np_rand2 = torch.rand(100, 100, 100, 100)
	rand = np.multiply(np_rand1, np_rand2)
	print("matmul completed on CPU")

# crunch_numbers_cuda()
# print("\n\n\n")
# crunch_numbers_cpu()

"""
This section lists some important tensor operations

torch.stack
torch.multinomial
torch.cat
torch.tril
torch.triu
<tensor>.T or <tensor>.Transpose
nn.Linear
F.softmax

"""

# torch.multinomial defines a probability distribution
# here [0.1, 0.9] means that there is 10% probablity of choosing index 0
# and 90% probabilty of choosing index 1
probabilities = torch.tensor([0.1, 0.9])

# now we will take 10 samples from the above probability space
# results will be different each time you run it.
# but it is given that if you run it enough times, you will see that 
# there is always 10% probability of random sample being 0 and 90% probability
# of random sample being 1
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
# print(samples)


# torch.cat concatenates two tensors
t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([6, 7])
t3 = torch.cat((t1, t2), dim=0)
# print("prints tensor [1, 2, 3, 4, 5, 6, 7]")
# print(t3)


# torch.tril gives lower triangular matrix
tril = torch.tril(torch.rand(5, 5))
# print(tril)
tril = torch.tril(torch.ones(5, 5))
# print(tril)

# torch.triu gives upper triangular matrix
tril = torch.triu(torch.rand(5, 5))
# print(tril)
tril = torch.triu(torch.ones(5, 5))
# print(tril)

# application of tril and triu is in exponentiation
out = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0, float('-inf'))
# print(out)
# print(torch.exp(out))

# <tensor>.Transpose tranposes a matrix. Rows to columns and columns to rows
# simple 2d example
t = torch.rand(3, 1)
r = t.T
# print(t, "\n\n", r)
# print(t.shape, r.shape)

# multi dimensional example
t = torch.zeros(4, 5, 6)
r = t.transpose(0, 2)	# swap 0th index with 2nd index
# print(t.shape, r.shape)

# torch.stack just stacks tensors
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([1, 2, 3])
t3 = torch.tensor([1, 2, 3])
t4 = torch.tensor([1, 2, 3])
stacked = torch.stack([t1, t2, t3, t4])
# print(stacked)


# nn.Linear creates a linear layer in NN definition
# it a part of torch.nn module
# this module contains everything with learnable parameters
# ie it contains all the different layers
sample = torch.tensor([10., 10., 10.])
linear = nn.Linear(3, 3, bias=False)
# print(linear(sample))

# torch.functional contains relu, tanh softmax etc.
# so
t1 = torch.tensor([1.0, 2.0, 3.0])
softmax_output = F.softmax(t1, dim=0)
print(softmax_output)

