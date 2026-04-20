---
title: "Training a Basic Language Model (Trigram Language Model)"
description: "Training a trigram language model from scratch — using both the counting approach and a gradient-based neural network approach in PyTorch."
pubDate: 2024-09-18
readTime: "13 min read"
tags: ["language-models", "trigram", "pytorch", "neural-networks"]
---

Large Language Models (LLMs) have revolutionized artificial intelligence in my opinion. When I first explored OpenAi's GPT 3.5 Turbo two years back, I was shocked. That how this models works. Then with the passage of time, other LLMs like Google's Gemini, Anthropic's Claud, Microsoft's Copliot and many more amazed us with their increased context lengths, increased parameters, and accuracy as well as RAG in these models. This advancement sparked curiosity inside me, that how these giants language models work in the backend? How do they predict things? What's inside there? Which kind of mathematics and statistics is involved? So, to quench my thirst, I decided to go deep down to the very basics and understand from there, that how language models are built, how mathematics is involved, how to train them, how to increase their accuracy so they predict accurately.

In this blog post, I will train a trigram language model. Trigram means three English alphabets, and a trigram language model predicts a next alphabet based on two previous alphabets. For example, it predicts the next alphabet with a highest probability, and this probability is dependent on the two consecutive immediate alphabets right before the predicted alphabet. Which means that the two alphabets is the input and third predicted alphabet is the output. So, our model will get a training set, it will learn from it, and then it will predict next alphabets. The predicted output won't make a sense, because it is just a basic level language model which will act as a step towards understanding complex language models.

I will also explain some technical things as well, such as the importance of dimensions of arrays and tensors in training such model, how broadcasting is important in the accuracy of model, how to save ourselves from hidden bugs, how to accurately use mathematics, and the optimization of model via gradient descent. I will train the model in two ways, first will train by counting methods manually, and secondly we will give our input to the neural network. I will extensively use Pytorch framework, and basic understanding of math and python is important to understand this.

The complete code notebook as well as input file is given in the link below, I will discuss important chunks of code below. So, lets begin.

## Counting Approach

We will upload our input data file which in .txt format which contains around 32,000 names. After uploading first we will check how many trigrams are there and how many times they had occurred.

```python
dict = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for chr1, chr2, chr3 in zip(chs, chs[1:], chs[2:]):
    Trigram = (chr1, chr2, chr3)
    dict[Trigram] = dict.get(Trigram, 0) + 1
sorted(dict.items(), key = lambda kv: -kv[1])
```

Here, `words` is our input file, and `w` represents alphabet. We create a list `chs` which has `<S>` as a starting alphabet, a list of all alphabets and then ending alphabet `<E>`. The `for loop` takes three consecutive alphabets from `chs` which makes one trigram, and then we store all the trigrams in the dictionary which gives us the counts of each bigram separately.

![Counts of each trigram sorted by frequency](/blog/training-a-basic-language-model-trigram/image_1.png)

The first trigram shows that names: an input files, has 287 names end with (`i`, `s`). `l` comes after (`a`, `e`) 287 times, and there are 284 names which start from (`m`, `e`) which is the fifth trigram in the above picture. These counts are very important, we will build probability distribution from this that will help us predict our output.

Now, we will create a tensor of 3D, because we are dealing with three alphabets at a time, two alphabets of input and one the output alphabets. And we have total of 26 alphabets, and there is one start and end character. To save ourselves from `S` and `E`, we will replace them with just `.` and now we have total 27 characters.

```python
import torch
N = torch.zeros((27,27,27))
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
for w in words:
  chs = ['.'] + list(w) + ['.']
  for chr1, chr2, chr3 in zip(chs, chs[1:],chs[2:]):
    ix1 = stoi[chr1]
    ix2 = stoi[chr2]
    ix3 = stoi[chr3]
    N[ix1, ix2, ix3]+= 1
```

Basically, we are dealing with a machine learning model, which accepts numerical input rather than textual data. So, its important to convert textual data (characters/alphabets) to numerical indices and vise versa. So, `chars` extract the unique set of alphabets which in our case `a` to `z`, and one special token `.` which is a start and end marker. `{stoi - string to integer}` is a dictionary that maps each special character (string) to a unique index (integer). This numerical representation is what machine learning require. `{itos - inter to string}` maps indices (integers) back to the characters (strings). This will help us to interpret the output of a model which will be in numerical form. `ix1 = stoi[chr1]` and so on convert each character in the trigram to its corresponding index using the `stoi` dictionary. `N[ix1, ix2, ix3] += 1` increments the corresponding position in the 3D tensor `N` based on the indices of the trigram. Each element `N[ix1, ix2, ix3]` in tensor holds how many times the trigram (`chr1`, `chr2`, `chr2`) occurred in the data set of words (names).

Now, we have created a tensor which contains three dimensions, two dimensions contain the input characters indices and one dimensions contains the index of output characters index. This array gives us the counts of each trigram. Now we will convert these counts into the probability distribution.

```python
P = (N+2).float()
P = P / P.sum(1, keepdims = True)
P.sum(1, keepdims = True).shape

# output
torch.Size([27, 1, 27])
```

Here, we have to do smoothing before calculating the probabilities. In `N`, some trigrams might not appear in our training data, so certain entries in `N` will be zero. This causes issues because probabilities based on trigram counts would also be zero for those unseen trigrams. By adding a small number, no entry is now zero and model assign those unseen trigrams a small probability. This also improves model's generalization.
Here, most importantly, we have to use `p.sum(1, keepdims = True)`, because if we use just `p.sum()`, it will squeeze the dimensions of the tensor, along second dimension and it will have shape of single scaler value (27). So shape of `P` is (27,27,27), and `p.sum()` will be (27). So, these two tensors cannot be broadcasted accurately. This will work and will show no error but dimensions will be reduced and we need three dimensions. NOTE: This is the hidden bug. So `p.sum(1, keepdims = True)` will save dimensions for us, and this broadcasting will give us a probabilities tensor of (27,27,27).

Now, we have trained a model, we will just draw a predicted sample of strings or characters from the probability distribution.

```python
g = torch.Generator().manual_seed(21475)

for i in range(5):
    out = []
    ix1, ix2 = 0, 0

    while True:
        p = P[ix1, ix2]
        ix  = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        ix1, ix2 = ix2, ix
        if ix == 0:
            break

    print(''.join(out))
```

Here, we have used a generator with a specific seed, which controls the randomness. So, by documentations, if the same seed is used, same output will be gotten. But, now a days Pytorch generator changes the output, and it depends on which CPU or GPU your using it. So, its possible that it can give different output. Here, the `ix1` and `ix2` is initiated with zero. These are the first two alphabets which is the input. `P[ix1, ix2]` is the 2D array which probabilities will predict the third alphabet. Because we have used `torch.multihnoimial` which draws sample from the probability distribution of the third alphabet. `torch.multihnoimial` only accepts input of either 1D or 2D, so either `P` has to be flattened or can be used the way I did. `ix` is the third alphabet, and it is then converted back from a numerical value via `itos` dictionary. We will get five samples in a joint string, which would be the predicted strings, which the model learned from the input `names` file. So, this is how basic language models are trained and they predict output based on probabilities.

Now, after training a model, we have to calculate the loss of this model. The more less the loss is, the more accurate the model assigns probabilities to the predicted alphabets.

```python
Log_liklihood = 0.0
n = 0
for w in words[:5]:
  chs = ['.'] + list(w) + ['.']
  for chr1, chr2, chr3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[chr1]
    ix2 = stoi[chr2]
    ix3 = stoi[chr3]
    prob = P[ix1, ix2, ix3]
    log_prob = torch.log(prob)
    Log_liklihood += log_prob
    nll = -Log_liklihood
    n += 1
print(f'{Log_liklihood=}')
print(f'{nll=}')
print(f'{nll/n=}')
```

Our model predicts a probability distribution over possible outcomes (for example, predicting the next character in a sequence). These predicted probabilities reflect how confident the model is that each possible outcome (e.g., next letter) will occur. `prob = P[ix1, ix2, ix3]` gets the predicted probability for the trigram (`chr1`, `chr2`, `chr3`) from the `P` tensor, which holds the probability values for all possible trigrams. We have to add these probabilities to get the overall accuracy of a model. In technical terms we have to calculate the maximum likelihood estimation (MLE). We take log of these probabilities because working with probabilities in log form is numerically more stable and easier to accumulate when multiplying many small numbers (which is what happens when calculating probabilities over sequences). So, log probabilities are added to the total log likelihood or MLE. So, we aim to maximize the log likelihood, the higher the likelihood, the more accurate the model is predicting. For connivence, we multiply it by negative one, so we minimize the negative log likelihood (`nll`), which is the overall loss of our model. So, we minimize the loss, the more less the loss is, the more accurately a trigram model is predicting or giving accurate probability to the next alphabet.

So, this was the basic training of a language model, which predicts the next character based on the probabilities of the previous two characters (bigram). We drew a predicted sample as well as calculated a loss of this model.

## Gradient Approach

To understand the gradient approach and how basic neural networks are designed, trained, and optimized, I would highly recommend to refer to my previous blog post: [Previous Blog Post](/blog/calculus-derivatives-in-neural-network-training/)

Here we will pass our input to the neural network, that will process the input, having weights, and will give us the predicted output.

```python
# now creating a training dat set for trigram model, gradient approach
xs, ys = [], []
for w in words[:]:
  chs = ['.'] + list(w) + ['.']
  for chr1, chr2, chr3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[chr1]
    ix2 = stoi[chr2]
    ix3 = stoi[chr3]
    print(chr1, chr2, chr3)
    xs.append([ix1, ix2])
    ys.append(ix3)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

Here, we create the training set `xs` which contains the input two characters (bigram) and `ys` is our predicted set which is the third character of a bigram we want the model to predict. These two are then converted into tensors because tensors are relatively easy to work with. `xs` and `ys` can look like below:

![xs and ys tensors for a single word from the training set](/blog/training-a-basic-language-model-trigram/image_2.png)

Above example is just for one word from our training set. `xs` has 4 sequences and each sequence contains two characters. For instance, `[0, 5]` is the index of two input alphabets and it want to predict the third alphabet which index is on `[13]` in the predicted tensor. These are just the indices of the letters not the probabilities. Now, we will pass these indices in `xs` of entire input set to the neural network. But neural network doesn't take strings, just labels or indices as an input. We have to convert each index to a vector.

```python
# one_hot encoding
xenc = F.one_hot(xs, num_classes=27).float()
num_sequences = xs.size(0)
xenc_flat = xenc.view(num_sequences, -1)
#Creating a neuron
W = torch.randn((54, 27), requires_grad=True,  generator=g)
logits = xenc_flat @ w # These are log counts
counts = logits.exp() # This is equivelant to N in our Previous method.
prob = counts / counts.sum(1, keepdims = True) # These are probablity distribution
log_prob = torch.log(prob) # converting probablities to log
```

We first convert the input data into one-hot encoded vectors, where each character is represented as a unique vector. `num_classes = 27` shows that we have total 27 unique characters. Next, `xenc_flat = xenc.view(num_sequences, -1)` takes the one-hot encoded vectors (which could be multi-dimensional) and flattens them into a 2D array. Each row corresponds to one sequence, and all the one-hot encoded values for that sequence are combined into one long row. The `-1` means "flatten the rest of the dimensions into this," so it simplifies everything after the first dimension (which is `num_sequences`). Because, we are building a neuron of size `(54, 27)`, so for broadcasting its necessary for `xenc` to match the size of `w`. 54 means that we have two input characters (27*27) and 27 shows the dimension of our predicted letter. When we multiply the input with this weight matrix, we get raw outputs called `logits`, which represent log counts. To make sense of these logits, we apply an exponent function to get counts (similar to how we counted trigrams before: `N` Tensor). We then normalize these counts into probabilities by dividing by the sum of all counts, so we get a probability distribution. Finally, we take the logarithm of the probabilities, which helps with the loss calculation. This is basically the softmax activation function. Here our model is ready.

```python
neg_log_liklihood = torch.zeros(4)
for i in range(4):
  x1, x2 = xs[i].tolist()
  y = ys[i].item()

  print(f'trigram example {i+1}: {itos[x1]}{itos[x2]}{itos[y]} (indexes {x1},{x2},{y})')
  print(f'input to the neural net:', x1, x2)
  print(f'output probablities from the neural net', prob[i])
  print(f'lable (actual next character)', y)
  p = prob[i, y] # Probablity of the correct next charater
  print(f'probablity assigned by the neural net to the correct charater', p.item())
  logp = torch.log(p) # log probablity
  print(f'log liklihood', logp.item())
  neg_log_liklihood[i] = -logp
  print(f'negative log liklihood:', neg_log_liklihood[i].item())
  #neg_log_liklihood[i] = neg_log_liklihood

print('=========')
print('average negative log liklihood:', neg_log_liklihood.mean().item())
```

The above code is all together of what we have done above. It will predict the loss of 4 iteration, which is the negative likelihood or MLE, we have calculated in the counting part as well.

Now, we have to optimize our model. First, there will be a forward pass, a model will calculate a loss, then there will be a backward pass, and then we will update the weights in the direction to minimize the loss.

Note: Please refer to the blog post link given above to know how forward pass, backward pass, and updating of weights work in the backend to have a clear idea.

```python
for k in range(10):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=27).float()  # One-hot encode
    xenc_flat = xenc.view(num_sequences, -1)      # Flatten the last two dimensions
    logits = xenc_flat @ w                        # Matrix multiplication
    counts = logits.exp()                         # Exponential to get counts
    prob = counts / counts.sum(1, keepdim=True)   # Normalize to get probabilities

    # Calculate loss
    loss = -prob[torch.arange(num_sequences), ys].log().mean() + 0.01*(w**2).mean()
    print(f'Loss: {loss.item()}')

    # Backward pass
    w.grad = None                                 # Zero out gradients
    loss.backward()

    # Update weights
    with torch.no_grad():
        w += -10 * w.grad                         # Gradient descent step
```

In the above code there are total 10 iterations which can vary, there is a forward pass which calculate the loss based on probabilities. `0.01*(W**2).mean()` is called regularization or smoothing we did in counting method as well. `num_sequences` are the total number of rows of bigrams (the input charaters) in our input. Then there is a backward pass where gradients are set to zero (explained in linked blog post) so that gradients don't get accumulate in each iteration. Then in the end there is an updating of weights in the opposite direction of gradient and in the direction of decreased loss.

We can also draw a sample of predicted strings or output characters just like we did in the counting method.

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix1, ix2 = 0, 0  # Start with 0 for the first two characters

    while True:
        xenc1 = F.one_hot(torch.tensor([ix1]), num_classes=27).float()
        xenc2 = F.one_hot(torch.tensor([ix2]), num_classes=27).float()
        xenc = torch.cat((xenc1, xenc2), dim=1).view(1, -1)  # Shape (1, 54)
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        # Sample the next character from the probability distribution
        ix2 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix2])
        if ix2 == 0:
            break

    print(''.join(out))
```

## Final Remarks

This training was just a basic level language model to get an intuitive idea of how these language models are trained, how they predict outputs based on probabilities. And how mathematics is involved in it. This training also gives us an idea of how dimensions of tensors are important. Without proper initialization of tensors with proper dimensions according to the requirements, can cause error in broadcasting. The outputs you get will not be understandable, because its just a one layer neural network. But the working is same to train multilayer neural networks with understandable outputs.

This blog got a bit longer, I hope it will be helpful. I will bet you to do the same working and train a model which takes three characters as an input and predict the fourth output characters, and the compare the loss with trigram model.

Share with me as well :p

[Code Notebook →](https://colab.research.google.com/drive/1e4KlENAy3uMp-SKA1hYTVztUKVNXwPMH?usp=sharing#scrollTo=TaYjHA86oze9)

[Input Data File →](https://drive.google.com/file/d/1tJFmlLpbFdl9YY_mPtaKs48denAP3Ibb/view)
