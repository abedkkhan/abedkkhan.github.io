---
title: "How to correctly Initialize the Neural Network: Mechanistic Interpretability Part 1"
description: "Why the last layer of a neural network should be initialized near zero — and how this small change dramatically improves training dynamics."
pubDate: 2024-12-21
readTime: "10 min read"
tags: ["mechanistic-interpretability", "neural-networks", "initialization", "pytorch"]
---

![Cover image — correctly initializing a neural network, mech-interp part 1](/blog/correctly-initialize-neural-network-mech-interp-1/cover.svg)

Artificial Intelligence's research and new models with more advance architectures is evolving very rapidly. But at the same time research dynamics are shifting towards analyzing and understanding the more hidden bugs in training neural networks. The research is trying to understand neural networks by breaking them down into more smaller and understandable parts. The goal is to understand each smaller part and how these smaller parts interacts with each other to make up the entire behavior of a neural network. It also identifies the hidden bugs during initialization, training, and optimization. Bugs, which we might unconsciously ignore and can cause a problem in network training and learning patterns. This whole phenomena is called "Mechanistic Interpretability".

So, basically, now to get better output, you don't just have to change your input, change the number of parameters, hyperparameters, embedding dimensions, activation functions or another thing. You can also, and this can have a better impact on over all training, inspect the network through the lens of "Mechanistic Interpretability".

In this blog post, we will train a trigram language model, and we will inspect how with just a small step we can accurately initialize our network, without changing any other thing, and at backend, that can create a significant impact. We will use a small example, but this can be applied to more larger and complex neural networks with hundreds of hidden layers and millions of parameters. Idea will be the same, just increase the size of your network, and you will see the change. so lets begin.

Note: To check how simple Trigram Language model is trained, refer to my [other blog post](/blog/training-a-basic-language-model-trigram/), in this blog post I will directly initialize the network.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
%matplotlib inline
words = open("names.txt", 'r').read().splitlines()
```

We import Pytorch and our input names file. And since we have total of 26 alphabets plus one special character `.`, so total we have 27 characters. We build vocabulary of our characters and mappings to/from integers just like below.

```python
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)
```

We split the data into train, test, and evaluation dataset just like below.

```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
```

Now, we will begin our initialization. We will embed each one of our 27 characters into 10 dimensions and will have 200 neurons in our hidden layer. And since our vocab size is 27, and embedding dimensions are 10, so our embedding matrix `C` will have (27, 10) dimensions. Our hidden layers `W1` will have dimension (10*3, 200) because we are taking three characters each has 10 dimensions, and we want to convert these into 200 dimensions to capture the most underlying patterns. The output of this layer will be the input to our last layer which is also called softmax layer, because softmax function convert raw logits into probability distribution. So, its dimension will be (200, 27). The output of our final layer is 27, because we have total of 27 characters and our output will be comprised of those characters samples. Below code explains this.

```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
#-----------------------------------------------------------------------------------
# Hidden layer
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) 
b1 = torch.randn(n_hidden,                        generator=g) 
#-----------------------------------------------------------------------------------
# Last (softmax layer)
W2 = torch.randn((n_hidden, vocab_size),          generator=g) 
b2 = torch.randn(vocab_size,                      generator=g) 



parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Here, we randomly initialize the entire network, including last layer. The last layer produce the logits which are then converted into probability. Here, the last layer is not centered around zero, or any other small number, so the logits it will produce will have a high magnitude. So, technically, softmax will produce a non-uniform probability distribution. Here, the network has not learned anything meaningful but it will give a very high initial loss and that might make the optimization harder to navigate. Because, the network, is falsely highly confident towards predicting certain classes.

So, during training, the optimization and during backpropagation, weights are adjusted to minimize the loss. If the initial loss is large in magnitude, the optimizer will squash these values down to achieve a better probability distribution. So larger logits can cause the model to be confidently wrong, increasing the initial loss and the optimizer has to work hard to adjust these larger logits to more reasonable values, which can slow down the entire training.

```python
# Optimization
from math import log

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # Minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward Pass
    emb = C[Xb] # embedd characters into vectors
    embcat = emb.view(emb.shape[0], -1) # Concatenate the vectors
    # linear layer
    hpreact = embcat @ W1 + b1
    # Applying non linearity
    h = torch.tanh(hpreact) # hidden layer
    # Output (softmax) layer also called logits (raw output)
    logits = h @ W2 + b2
    # Loss function
    loss = F.cross_entropy(logits, Yb)

    # Backward Pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    lr = 0.1 if i < 100000 else 0.01 # learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # Track Stats
    if i % 10000 == 0: # print every once in a while
       print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
```

The loss we get after doing optimization like above is just like this.

![Loss output across training steps with random initialization](/blog/correctly-initialize-neural-network-mech-interp-1/image_1.png)

Below diagram can better explain this loss.

![Loss curve showing a hockey-stick pattern with very high initial loss](/blog/correctly-initialize-neural-network-mech-interp-1/image_2.png)

The loss we get after doing optimization like above is just like this. We can clearly see that initially we have a very higher loss and then slowly optimizer adjusts it. Initially, the network is wrongly confident
in prediction of some patterns or classes. Which is not feasible for our network training. If we evaluate
this network on test and dev data set, we get loss like below.

```python
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]

  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')
split_loss('test')

#output
train 2.122785806655884
val 2.1687753200531006
test 2.157045364379883
```

And more interestingly, when we finally draw a sample from predicted classes, we get something like
below.

![Sampled names generated from the poorly initialized network](/blog/correctly-initialize-neural-network-mech-interp-1/image_3.png)

Indeed, this is a toy example, it won't predict accurate names but the classes or names it has predicted are still very wired. So, it means that there is some issue. Now, we will reinitialize our network and we will reinitialize our last Softmax layer near zero, and will compare and see how we get uniform logits, uniform probability distribution, less and reasonable initial loss, and how our model wrong confidence towards certain classes will be decreased.

## Reinitialization of our Network

```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) 
C  = torch.randn((vocab_size, n_embd),            generator=g)
#-----------------------------------------------------------------------------------------
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) 
b1 = torch.randn(n_hidden,                        generator=g) 
#-----------------------------------------------------------------------------------------
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0
b2 = torch.randn(vocab_size,                      generator=g) * 0



parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Here, we can see that, everything is the same, but we have multiplied last layer by zero. So, we will do some mathematics. Our last layer, which computes a set of raw score for each class, and then these logits are passed through Softmax to give probability distribution. So mathematically it looks like:

![Softmax equation producing uniform probability when logits are zero](/blog/correctly-initialize-neural-network-mech-interp-1/image_4.jpeg)

And we get equally likely probability for all classes, and then we apply Cross Entropy loss function to our probability distribution and we will see that we get our initial loss pretty reasonable.

![Cross entropy calculation yielding a reasonable initial loss](/blog/correctly-initialize-neural-network-mech-interp-1/image_5.jpeg)

We got a pretty reasonable initial loss, and clearly we can see that model at the early stage is not wrongly confident towards certain classes, and now when we optimize the model again , we will see the results as below.

![Loss output across training steps after proper initialization](/blog/correctly-initialize-neural-network-mech-interp-1/image_6.png)

![Smooth loss curve — no hockey stick after last layer is initialized near zero](/blog/correctly-initialize-neural-network-mech-interp-1/image_7.png)

There is no extreme high initial loss in the start, the optimizer is not squashing these values down to achieve a better probability distribution and we get a non skewed loss distribution. Also, we don't have any hocky stick distribution, the network is adjusting its weights with a reasonable learning rate, and it can save our time in training for more complex MLP.

```python
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]

  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')
split_loss('test')

# output
train 2.0698275566101074
val 2.130896806716919
test 2.132791519165039
```

As compared to our previous case, our test and dev loss is less as well, which means that accurately initialization of our last layer has an impact. Now, if we draw samples from predicted classes, they will look more reasonable just like below.

![Sampled names after proper initialization — more reasonable outputs](/blog/correctly-initialize-neural-network-mech-interp-1/image_8.png)

As Names are still not accurate, but they do not look wired as compared to our previous case.

## Conclusion

So, in conclusion, this was a toy example, and we don't see clear difference between our final losses in both the cases. But, if we look from Mechanistic Interpretability lens, we can clearlysee that how last layer gives us output, and how that create a non uniform probability distribution. And that makes the model to be very confident but wrong in predictions of certain classes because it gives us very high skewed and high initial loss. And when we pull the last layer down by a gravitational force caused by multiplication of zero, we get uniform logits, and reasonable initial loss, and we make the last layer less confident. In more complex model, this small but important step can save a lot of time for training and optimization, and we get pretty descent predictions just like we saw from the drawn sample.

In the next blog post, we will look into how can we accurately initialize our hidden layer, and that can save our neural network from dead neurons. And we will see which activation functions are very vulnerable towards dead neurons in the neural networks.

Below is the link to complete code notebook.

[GitHub Repo →](https://github.com/abedkkhan/ML-topics-I-love-exploring/blob/main/How%20to%20correctly%20Initialize%20the%20Neural%20Network.py)
