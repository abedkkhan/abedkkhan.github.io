---
title: "The Kaiming initialization and Batch Normalization in Neural Networks"
description: "Moving beyond magic numbers — using Kaiming He initialization and Batch Normalization to formally stabilize neural network training."
pubDate: 2025-01-07
readTime: "14 min read"
tags: ["neural-networks", "kaiming-init", "batch-normalization", "initialization", "pytorch"]
---

![Cover image — Kaiming initialization and Batch Normalization](/blog/kaiming-init-batch-norm/cover.webp)

In [my last blog post](/blog/dead-neurons-mech-interp-2/), we discussed and solved through code, how to initialize the hidden layer in neural network properly, to get rid of dead neurons. We came up with some magic numbers, randomly, as well as, by looking at the mathematics of `tanh` activation function, such as `0.1`, `0.2`, and `0.01`. We multiplied these numbers with our hidden layer. And, luckily, these numbers worked and we eliminated dead neurons. But the question is, do these magic numbers will always work? What if we have different activation function and have a larger neural network with many hidden layers and millions or billions neurons? So, the answer is, that these magic numbers won't work always and we luckily, have formal and more modernize solution to this problem such as Semi-principled Kaiming Initialization and batch normalization.

The Kaiming initialization method also known as Kaiming He initialization or He normal initialization was first explained in a paper [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852) back in 2015. This is the formal technique for initializing the weights of neural networks. The aim of this technique is to address the issue of vanishing or exploding gradients or the dead neurons. This method is calculated as a random number with a Guassian probability distribution having mean `0` and standard deviation = `gain/ fan-mod**0.5` where `fan-mod` is the input of a layer to which it is applied and the `gain` value varies for different activation functions such as for `tanh` the gain is `5/3` and for `ReLu` the gain is `2`. Checkout [pytorch docs](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) for different gain values for different activation functions. We won't dive into the underlaying mathematics of this, because fortunately we have pytorch as well as we know the formula, which we can apply directly. In pytorch there is a function `torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='', generator=None)`. Which can be directly used in a code to apply this method. Also, we can use the formula as well. In this blog post we will go with the latter.

Batch normalization was first discussed in a paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167) back in 2015 also. Batch normalization is a powerful and modern technique primarily used to stabilize and accelerate neural networks training. Basically, in neural networks, when data passes through layers, each layer input distribution changes. This change is called covariate shift and it can make training less stable and slower. By normalizing inputs within each batch, batch normalization reduces this shift, and make a way for a network to learn more efficiently. Also, normalizing each layer input means to have a mean closer to zero and standard deviation closer to 1. This helps to keep the network gradients within a stable range and avoids issues like exploding or vanishing gradients or dead neurons. Batch normalization helps neural network converge faster and often allow higher learning rates, making training faster and stable.

Batch normalization does not require to do perfect math, it just takes care of the activation distributions for all these types of neural networks and it significantly stabilize the neural networks.

## Kaiming initialization method

Since in the [previous blog post](/blog/dead-neurons-mech-interp-2/) we multiplied those magic numbers with the weights and biases of our hidden layer because those weights and biases were involved in activation function. Now, in this case, we will multiply the standard deviation of Kaiming method with the weights of our hidden layer. Standard deviation = `gain/fan-mod**0.5` where gain is `5/3`. We will initialize our network in the same way as below.

```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
#--------------------------------------------------------
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / ((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
#--------------------------------------------------------
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0

parameters = [C, W1, b1 W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Here, we multiplied `W1` by the standard deviation of Kaiming method. The gain is `5/3` for `tanh` function and `n_embd * block_size` is the input of hidden layer. Which is also called `fan-mod`. This multiplication will restrict the activations distribution to be Gaussian, and we will have a descent statistics of all these activations. This will provide stability, we will eliminate the dead neurons or exploding gradients in the backpropagation. And the whole network will learn effectively. Now, we optimize this network and we get descent initial loss, but we are more interested in dead neurons which is shown below.

```python
from math import log
max_steps = 200000
batch_size = 32
lossi = []
for i in range(max_steps):

   # Minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # Forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear Layer
  hpreact = embcat @ W1 + b1 # hidden layer pre-activation
  # Non Linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad
  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  break
```

After optimization, we inspect dead neurons graphically as below.

```python
#used semi principled "kaiming init"
plt.hist(h.view(-1).tolist(), 50);
```

![Histogram of h after Kaiming initialization — values spread between -1 and +1](/blog/kaiming-init-batch-norm/image_1.png)

To, check the bar graphs, where we didn't multiply hidden layer weights by anything, please refer to the [previous blog post](/blog/dead-neurons-mech-interp-2/). In that blog post, it is explained that when we initialize hidden layer randomly, our bar graph of `h` which is activation function, has all the values on extremes such as `-1` and `+1`. But here in this case, we still have values at the extremes, but we also have values in between these two extremes. It means, this formal method of eliminating dead neurons, which is Kaiming initialization method, works fine. The below barcode type graph tells the same story.

```python
#used semi principled "kaiming init"
plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray' , interpolation='nearest')
```

![Barcode plot after Kaiming initialization — mostly black, few dead neurons](/blog/kaiming-init-batch-norm/image_2.png)

The black region shows those neurons which remain active and learn while white region shows dead neurons or vanished gradients. We can clearly see that after Kaiming initialization method, we have more black region, means most of the neurons are learning and contributing to the final loss.

In conclusion, we saw that there are formal and authentic and more generic ways such as Kaiming initialization method which we can use in our network initialization. More technically, proper initializing of hidden layer[s] which is directly linked with the activation functions and we don't have to worry about the mathematics of activation function in the backpropagation and their derivatives. But, it is advised to do some math before following any method. And we can precisely initialize and train our network, to keep our gradients in a good range and save them from exploding or vanishing.

## Batch Normalization

When training neural networks, it is important to manage the pre-activation states (the values before applying non-linear activation functions like `tanh` or `ReLU`). If these pre-activation states are too small, the activation functions fail to contribute effectively to learning—for instance, the `tanh` function outputs values close to zero, losing its ability to distinguish between inputs. Conversely, if the pre-activation states are too large, activation functions saturate, producing outputs that are constant for a wide range of inputs. This leads to the "dead neuron" problem, where neurons stop learning because gradients vanish. Ideally, we want these pre-activation states to resemble a Gaussian distribution with a mean of `0` and a standard deviation of `1`, at least during initialization, as this facilitates smoother gradient flow and efficient learning.

This is where batch normalization comes into play. Batch normalization makes neural network training more flexible by actively ensuring that hidden states (or pre-activation states) are normalized to have desirable statistical properties. Instead of relying solely on careful initialization to achieve Gaussian-like behavior, batch normalization takes the hidden states during training and dynamically normalizes them to follow a Gaussian distribution. This not only addresses the issue of exploding or vanishing activation values but also accelerates convergence, making the training process more robust to hyperparameter settings.

![Batch Normalization algorithm from the original 2015 paper](/blog/kaiming-init-batch-norm/image_3.png)

Above is the screenshot of an algorithm from the [paper](https://arxiv.org/pdf/1502.03167) which introduced batch normalization. The batch normalization is applied on a hidden or pre-activation states. In more easy words, on the input of a hidden layer, which is considered as a mini batch. The algorithm first compute the mean of a mini batch of an input data, then a standard deviation, and then it normalize the entire mini batch. But the last step, scale and shift is a bit tricky. Scale and shift are the two parameters, algorithm learns to adjust the normalization.

So, basically, in Step 3, we already have the normalized values, which are centered around `0` (mean = 0) and scaled to have unit variance (standard deviation = 1). This ensures the data is well-behaved for training. Now, in step 4, the algorithm perform two additional operations on these normalized values: scale and shift which are learnable parameters. The scale changes the range of normalized values to scale up (stretch) and scale down (compress). While, shift, moves the normalized values up and down. For example, if shift value is `2`, all the values shift `2` units upward.

So, the question is why do we scale and shift? So, Normalization (Step 3) standardizes the values, but it can remove some useful information about the original data's scale and offset. By introducing scale and shift, the network can learn the best range and position for the normalized values, if needed. Also, without these learnable parameters, the normalized values are strictly fixed to mean `0` and std `1`. But not all layers in a neural network benefit from strictly zero-mean and unit-variance data. Scale and shift allow the model to adjust the normalized data dynamically during training. Also, scaling and shifting ensure that the activations are in the range most suitable for the next layer's activation function (like `ReLU` or `tanh`). For example, `ReLU` works best when inputs are non-negative, and `tanh` works best when inputs are spread across its range. A good part is that, a network learns scale and shift as part of the training process. Which means that the network itself decides the optimal scale and offset for the activations, adapting to the specific problem it is solving.

For example, in layman terms, suppose we are normalizing the heights of students in a classroom. We subtract the average height and divide by the standard deviation, so all heights are normalized. But now, what if we need to plot their heights on a specific scale, like in centimeters (not standardized)? So, here, scale adjusts the "spread" of heights to match the scale (e.g., scaling it back to centimeters) and shift adjusts the "starting point" or base height to match the range we want (e.g., aligning to the shortest student's height). In a neural network, scale and shift allow the model to learn this type of adjustment for activations and giving the model a control to optimize for the best representation of data. This added flexibility leads to faster training and often better results.

So, we will initialize our network with batch normalization parameters.

```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
#-----------------------------------------------------------------------------------------
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / ((n_embd * block_size)**0.5)
#b1 = torch.randn(n_hidden,                        generator=g) * 0.01
#-----------------------------------------------------------------------------------------
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0

# BatchNorm Parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))


parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Above, we can see that our network is initialized in the same way, but we have some additional batch norm parameters.

1. `bngain = torch.ones((1, n_hidden))` which is basically the scale we described above. Its shape `(1, n_hidden)` matches the number of hidden neurons in the layer (`200` in our case). Each hidden neuron has a separate gain parameter that the network learns during training. This lets the network control how much to scale the normalized output for each hidden neuron.

It is initialized with ones because after batch normalization, the activations are normalized to have a mean of `0` and a standard deviation of `1`. This can change the scale of the activations. By initializing `bngain` to `1`, we ensure the activations are initially not scaled up or down, preserving their normalized range. During training, the network learns the optimal values of `bngain` to scale the normalized activations for each neuron. For example, if a neuron needs a stronger signal, `bngain` can increase (e.g., `>1`); if it needs to be dampened, `bngain` can decrease (e.g., `<1`). So, starting with `bngain = 1` means the scaling operation doesn't affect the network initially. It acts as a "neutral" initialization.

2. `bnbias = torch.zeros((1, n_hidden))` acts as a shifting factor after normalizing the pre-activation values.

Similar to `bngain`, this allows the network to learn an offset (or bias) for each hidden neuron. This makes the network more flexible by letting it adjust the mean of the normalized output.

It is initialized with zeros because batch normalization ensures the mean of the activations is `0` after normalization. By setting `bnbias` to `0`, we maintain this zero-centered mean initially. During training, the network learns the optimal values of `bnbias` to shift the normalized activations for each neuron. For example, if a neuron needs its mean to be higher, `bnbias` increases; if it needs to stay low, `bnbias` decreases. Initializing `bnbias = 0` ensures no unnecessary shifting of the activations initially, keeping the network's behavior consistent at the start.

The combination of `bngain = 1` and `bnbias = 0` makes the batch normalization layer behave like a "pass-through" initially (aside from normalizing the activations). This avoids introducing any unnecessary biases or distortions in the network at the start of training, allowing the network to adapt these parameters over time.

`bnmean_running = torch.zeros((1, n_hidden))` and `bnstd_running = torch.ones((1, n_hidden))` are running mean (`bnmean_running`) and running standard deviation (`bnstd_running`) update during training and serve as estimates of the overall dataset's statistics. Because, during training, use the mini-batch statistics (`bnmeani` and `bnstdi`) to normalize the data. But, in testing phase, we may not use mini-batches (or may want consistent behavior across different test inputs). Instead, we use the running mean and running standard deviation, which approximate the entire dataset's statistics. This ensures that the Batch Norm works consistently during training and testing.

Now, we are all set up, lets optimize our network.

```python
from math import log
# Same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

   # Minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # Forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear Layer
  hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation

  # BATCH Norm
  # -----------------------------------------------------------------------------------
  bnmeani = hpreact.mean(0, keepdim=True)
  bnstdi = hpreact.std(0, keepdim=True)
  hpreact = bngain * (hpreact - bnmeani) / bnstdi  + bnbias #hiddenlayer normalization
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  #-----------------------------------------------------------------------------------

  # Non Linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function


  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

  #break
```

A random minibatch of size `batch_size = 32` is selected from the training data (`Xtr` and `Ytr`). The minibatch `Xb` is passed through an embedding matrix `C`, converting input tokens (e.g., characters) into vectors of size `n_embd = 10`. The embeddings are flattened (concatenated across the sequence dimension) `embcat = emb.view(emb.shape[0], -1)`. The flattened embeddings are passed through the first linear layer (`W1`) to compute the hidden layer pre-activation values: `hpreact = embcat @ W1`. Here, one point to be noted is that we removed `b1` the bias term because batch normalization already removes the effect of biases. During Batch Norm, we subtract the batch mean (`bnmeani`) from `hpreact`. This effectively cancels out any bias term because the mean includes the bias. As a result, adding a bias term before Batch Norm is redundant and does not affect the output.

After calculating the pre-activation values (`hpreact`), BatchNorm is applied as follows:

a) Calculate the mean and std (for current batch):

The mean and standard deviation are computed across the batch dimension for each hidden neuron:

`bnmeani = hpreact.mean(0, keepdim=True)` `bnstdi = hpreact.std(0, keepdim=True)`

Shape of `bnmeani` and `bnstdi`: `(1, n_hidden)`, where `n_hidden = 200`.

b) Normalize the pre-activation values:

Each pre-activation value is normalized by subtracting the mean and dividing by the standard deviation:

`hpreact = (hpreact - bnmeani) / bnstdi`

This ensures that the activations for each hidden neuron have a mean of `0` and a standard deviation of `1` (Gaussian distribution).

c) Apply learnable scale (`bngain`) and shift (`bnbias`):

The normalized values are scaled and shifted using the learnable parameters `bngain` and `bnbias`.

`hpreact = bngain * hpreact + bnbias`

`bngain` (initialized to `1`): Controls how much to scale the activations. `bnbias` (initialized to `0`): Controls how much to shift the activations. These parameters are updated during training, allowing the network to adjust the distribution of activations as needed.

The running mean (`bnmean_running`) and standard deviation (`bnstd_running`) are updated using an exponential moving average:

```python
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```

`0.999`: Weight for the previous running value (gives it more importance). `0.001`: Weight for the current batch value (small influence). This ensures smooth updates to the running statistics over many minibatches. This running mean and std won't be involved in optimization and backpropagation and we will use it in our testing phase.

Hence, in this way, we initialize our network and apply batch normalization and then we optimize it. We get decent initial loss. But, the main problem, from which we started, is the dead neurons. Now, we will see, after applying batch normalization, do the problem of dead neurons still persists or not.

```python
# adding batch norm layer
plt.hist(h.view(-1).tolist(), 50);
```

![Histogram of h after batch normalization — most values lie within -1 and +1](/blog/kaiming-init-batch-norm/image_4.png)

We can see that, and if we compare the above plot with the plot we got in Kaiming initialization method, batch normalization did a more effective job. Most of the data, and more data as compared to Kaiming initialization method, lies within `-1` and `+1`. The batch norm has effectively improved the training and eliminated the problem of vanishing gradients or dead neurons up to greater extent. Below graph tells the same story.

```python
# adding batch norm layer
plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray' , interpolation='nearest')
```

![Barcode plot after batch normalization — mostly black, very few dead neurons](/blog/kaiming-init-batch-norm/image_5.png)

Since, we know that black region shows the neurons which learn during training and white region show the dead neurons. If we compare this barcode type graph with the one we got for Kaiming initialization method, its clearly visible that we have more black region. Means, the vanishing gradients or dead neurons problem is solved by batch norm up to maximum extent, and our network learned effectively.

Now, we will test and evaluate our trained network on the unseen data and here we will use the running mean and std, the network has calculated during the training.

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
  hpreact = embcat @ W1 #+ b1
  hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')
split_loss('test')

# results
train 2.0674147605895996
val 2.1056840419769287
test 2.1070175170898438
```

The training loss (`2.0674`) is slightly lower than the validation (`2.1057`) and test losses (`2.1070`), which is expected since the model optimizes directly on the training data. The close similarity between validation and test losses indicates good generalization, meaning the model performs consistently on unseen data. The small gap between training and validation/test losses shows the model is not overfitting and has learned meaningful patterns from the data. This stable performance suggests the validation process was reliable, and the model has likely converged to a good solution.

## Conclusion

In short, batch normalization improves accuracy and adds flexibility to training by addressing problems like dead neurons. While Kaiming initialization aims to keep the pre-activation values roughly Gaussian at the start, ensuring they aren't too small or large, batch normalization goes further. It normalizes the hidden states during training, making them roughly Gaussian, and it allows for scaling and shifting, meaning the mean and standard deviation don't have to be strictly `0` and `1`. This helps improve stability and training dynamics.

[GitHub code →](https://github.com/abedkkhan/ML-topics-I-love-exploring/blob/main/Batch%20Norm%20and%20Kaiming%20initialization.py)
