---
title: "Identifying and Removing Dead Neurons in Training Neural Networks: Mechanistic Interpretability Part 2"
description: "Diagnosing dead neurons caused by the tanh activation function and fixing them by properly initializing hidden layer weights."
pubDate: 2024-12-31
readTime: "12 min read"
tags: ["mechanistic-interpretability", "neural-networks", "dead-neurons", "activation-functions", "pytorch"]
---

![Cover image — dead neurons in neural networks](/blog/dead-neurons-mech-interp-2/cover.webp)

[My last blog](/blog/correctly-initialize-neural-network-mech-interp-1/) explained how to correctly initialize the last softmax layer of a neural network to reduce high but incorrect confidence when predicting certain classes. It also covered how to achieve uniform logits, a proper probability distribution, and a reasonable initial loss.

In this blog, we'll dive into the correct initialization of hidden layers during training. We'll also uncover a hidden bug called "Dead Neurons," explain it mathematically, and show how proper hidden layer initialization can save your network from this issue.

When we initialize our neural network, the next critical part is optimization, which has two main components: forward pass and backward pass.

In the forward pass, the network makes predictions and gives a loss value. The goal is to minimize this loss so the network can make accurate predictions. To minimize the loss, the network performs a backward pass, where the chain rule of calculus is used to calculate the partial derivatives of all weights with respect to the loss. Using these derivatives, the network adjusts its weights, performs another forward pass, calculates the loss again, and continues this process until the loss is minimized and predictions improve. Essentially, neurons keep learning and moving forward and backward through this process.

Another key component in this process is the activation function, which introduces non-linearity to the data. This non-linearity helps the network learn diverse and complex features from unseen data. However, activation functions also participate in the backward pass, as their derivatives are computed with respect to the loss, helping the network adjust weights and biases.

Here's the catch: the mathematical form of activation functions and their derivatives need close attention. If not handled properly, they can introduce hidden bugs during training, like "Dead Neurons", which can significantly hinder the network's learning. This is why the choice and initialization of activation functions are critical for stable and effective training.

First, we'll train the network without paying attention to how the hidden layer interacts with the activation function. This will help us observe how the mathematical form of the activation function, particularly after taking its derivative, can lead to the problem of dead neurons. Dead neurons are those that completely skip the training process—they don't update, don't learn, and remain "dead," just as the name suggests. They effectively stop contributing to the network's learning.

For this experiment, we'll use the `tanh` activation function. First, we'll initialize our network, run it, and see how dead neurons emerge due to improper initialization or the behavior of the `tanh` activation function. Let's dive into the details step by step.

See the [previous blog post](/blog/calculus-derivatives-in-neural-network-training/) of mine which explain the backward pass and forward pass and explains how chain rule and derivative is involved in network training.

```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
#-----------------------------------------------------------------------------------------
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) 
b1 = torch.randn(n_hidden,                        generator=g) 
#-----------------------------------------------------------------------------------------
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0



parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Great! Since we've initialized the network similarly to the approach in our previous blog post on mechanistic interpretability (Part 1), we already have a last softmax layer initialized by multiplying by `0.01` and `0`. However, for this experiment, we haven't scaled our hidden layer weights and biases (`W1` and `b1`), leaving them unadjusted.
Now, we'll proceed with optimization as usual:

1. Forward Pass:

. Compute the outputs of the network using the current weights and biases.

. Use the `tanh` activation function in the hidden layer.

. Calculate the loss (e.g., using negative log-likelihood or cross-entropy).

2. Backward Pass:

. Apply the chain rule to compute the gradient of the loss with respect to each weight and bias, working backwards through the network.

. Update the weights and biases using these gradients (e.g., with a learning rate and gradient descent).

3. Iterate:

. Repeat the forward and backward passes to continue reducing the loss.

By not scaling the hidden layer weights (`W1`, `b1`), we'll see how dead neurons emerge, especially when using the `tanh` activation function. This is because the derivative of `tanh` can shrink or saturate for certain input ranges, leading to zero gradients for some neurons—effectively making them "dead." Let's move forward to observe this in action!

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
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)

    # linear layer
    hpreact = embcat @ W1 + b1

    # Applying non linearity
    h = torch.tanh(hpreact)

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

    break
```

In the forward pass, the non-linear layer is computed as `hpreact = embcat @ W1 + b1` involving `W1` and `b1`. The `tanh` activation function is then applied, resulting in `h`. After completing all the forward and backward passes, we will visualize `h` graphically to observe how it behaves post-optimization.

```python
plt.hist(h.view(-1).tolist(), 50);
```

![Histogram of h values — majority clustered at -1 and +1](/blog/dead-neurons-mech-interp-2/image_1.png)

If we look `h` values are compact between `-1` and `+1` and more interestingly majority of the values are `-1` and `+1`. Why so? Because we have used `tan` activation function and since the range of `tan` is `-1` and `+1` mathematically. So, mathematically, these majority values has to be `-1` and `+1`. But, here we get the problem, we don't want the majority values `-1` and `+1`. Why we don't want it, will explain after looking at one another graph below.

```python
plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray' , interpolation='nearest')
```

![Barcode plot showing active neurons (black) and dead neurons (white)](/blog/dead-neurons-mech-interp-2/image_2.png)

The above barcode type graph of `h` tells us very interesting story. Both the white region and black region show neurons. Black region are those neurons which are active and learn in optimization, and white region are those neurons which don't learn in the optimization and remain dead and call dead neurons. So, now lets do some mathematics on `tan` activation function and see how it looks like and why it causes dead neurons.

```python
def tanh(self)
x = self.data
t = (math.exp(2*x)-1)/ (math.exp(2*x) + 1)
out = Value(t, (self,), 'tanh') # output of tanh

def _backward():
   self.grad += (1-t**2) * out.grad
out._backward = _backward
```

Above is the backend working of `tanh`. We see that in backward pass, everything depend on the value of `t`. If the value of `t` is `-1` or `+1`, there is a square in `t` and the whole value will of `self.grad` become zero, and that specific neuron will remain inactive and won't learn, and gradient (the optimizer) will just pass through it, so in this case some of the neurons will not learn and won't get optimal result and ultimately will remain dead. On the top, we see that our network gets train and we get a final loss. But under the hood, we have dead neurons, which don't even learn and do not contribute to the final loss and result. So, indeed, the weights and biases along `tanh` neurons do not impact the loss because the output of the `tanh` is zero, there is no influence. Gradients flowing through `tanh` can only ever decreased and the amount that they decrease is proportional through square `(1-t**2)` depends on how many values are `-1` and `+1`.

There is a simple and informal solution we will apply for our toy example, and in our next blog post we will solve this problem by applying a formal solution which is multiplication of hidden layer by a value of gain as well as batch normalization. Also, the case we discussed here can also be seen in other kinds of non linearities such as Sigmoid or Relu.

Here, in this case, mathematically we see that the values of `t` being `-1` or `+1` is causing an issue. We take a step back and see that `h` is made up from `W1` and `b1`. `h = torch.tanh(hpreact)` and then `h = torch.tanh(hpreact)`. So, we will go back to our network initialization step. And we will force the value of `t` to be not `+1` or `-1`. Since `h = embcat @ W1 + b1` is already Gaussian matrix, but `W1` and `b1` are initialized randomly, so technically they can give any values might be `+1` and `-1`. We multiply `W1` by `0.1` and `b1` by `0.01`, to some sort of restrict these values more closely to `0.1` and `0.01`. And this small magical step saves us from dead neurons, and don't get any values which makes the neurons bypass the `tanh` activation.

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.1
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
```

We initialize by making this change, we optimize, get a loss and now we inspect the `h` as below.

![Histogram of h after scaling W1 and b1 — values concentrated in the middle](/blog/dead-neurons-mech-interp-2/image_3.png)

We can see that now we don't have any extreme values on `-1` and `+1` and we forced all the values in the middle. Below graph also shows this change.

![Barcode plot after the fix — all neurons active, no dead neurons](/blog/dead-neurons-mech-interp-2/image_4.png)

There is no white region, all the regions is black. There are no dead neurons now, all the `tanh` neurons remain active, learn, adjust weights and biases and influence the final loss. But, machine learning experts like Andrej Karpathy says that we should have some diversity in our data. Means, we should have some dead neurons as well, less amount, which don't make any significant problem. So, if we multiply hidden layer `W1` by `0.2` we get some values `-1` and `+1` and makes some of the neurons to be dead. Graphically we can see below.

![Histogram of h when W1 is multiplied by 0.2 — some values at the extremes](/blog/dead-neurons-mech-interp-2/image_5.png)

We can see that still we have majority of the values between `-1` and `+1` but we also have some values at the extremes as well. Which will make some neurons dead, as shown in the below graph.

![Barcode plot with W1 * 0.2 — mostly active neurons with a few dead ones](/blog/dead-neurons-mech-interp-2/image_6.png)

We still have most of the black region, active neurons but at the same time we got some of the white region, the dead neurons.

## Conclusion

Most of the time we see that our network behaves well but under the hood there are many bugs which get ignored, cause problems mostly when training larger neural networks. So, we have to inspect each and every part of forward pass and backward pass, mathematically. Sometimes we have to solve all the mathematics on the paper to check what we get actually. Also, graphical inspection just like we did also helps a lot and that tell us very different story which we might not see by just looking at the code. This can save us from a severe problems. Just like in this case, we just saw the mathematical form of `tan` activation function, its derivate, we saw it graphically and then solve it.

But the issue is not solved yet, the question is, how did we come up with these magical numbers and we instantly multiplied by `W1` and `b1` and eliminated the problem. What if we have a larger network and different activation function. Will these magical numbers help?. The answer is not always. And we have more formal solutions to such problem. One is, each activation function has its gain value. And we have to multiply hidden layer by that gain value in a specific way (formula). Also, there is another modern technique which is called batch normalization. That approach gives us a great flexibility and solves a lot of such kind of hidden bugs. We will apply these approach in our next blog on the same network.

[GitHub Repo →](https://github.com/abedkkhan/ML-topics-I-love-exploring/blob/main/Identifying%20and%20Removing%20Dead%20Neurons.py)
