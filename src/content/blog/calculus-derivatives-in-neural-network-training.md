---
title: "How basic concept of Calculus (Derivative) has a Key Role in Training Neural Networks?"
description: "Walking through how derivatives, the chain rule, and gradient resets shape neural network training — built from scratch."
pubDate: 2024-09-05
originalUrl: "https://aabidkarim.hashnode.dev/how-basic-concept-of-calculus-derivative-has-a-key-role-in-training-neural-networks"
readTime: "9 min read"
tags: ["neural-networks", "calculus", "backpropagation", "training"]
---

A Neural Network is a machine learning model that works like the human brain. Just like how our brain has neurons that send signals, a neural network has artificial neurons that pass information. The network takes input, does some math, and passes the information to the next neurons until it reaches a decision.

Each neuron has a weight, which shows how important it is for that neuron to pass the signal. This is like how our brain gives more attention to important thoughts or emotions. When the network doesn't make the right decision, it uses a method called backpropagation to go back and check how much each neuron's importance affected the decision. Then, it adjusts these weights to try and get a better result next time. This process helps the neural network learn and improve over time, just like how we learn from our mistakes. Below is the basic structure of a neural network.

![Basic structure of a neural network](/blog/calculus-derivatives-in-neural-network-training/image_1.jpeg)

Adjusting weights in a backpropagation is very important to reach to the accurate decision (output). Partial derivative of output which is our final decision is taken with respect to the other neurons or leaf nodes or child nodes. This partial derivative is called gradient. Below is the simple three input of a neural network, two equations which are inner nodes and a final output L. The main player to compute these gradients is the chain rule from calculus.

```text
a = 2.0, b = -3.0
c = 10.0, f = -2.0
e = a * b, d = e + c
L = d * f

L = f * d
dL/df = d
dL/dd = f
d = c + e
dL/dc = (dL/dd) * (dd/dc)
dL/de = (dL/dd) * (dd/de)
e = a * b
dL/da = (dL/de) * (de/da)
dL/db = (dL/de) * (de/db)

a.grad = 6.0,  d.grad = -2.0
b.grad = -4.0, c.grad = -2.0
f.grad = 4.0,  e.grad = -2.0
L.grad = 1.0
```

![Computation graph showing forward and backward propagation of gradients](/blog/calculus-derivatives-in-neural-network-training/image_2.png)

The above graph shows how inputs propagates through mathematical expressions and gives the final output, and how in backpropagation gradients are calculated.

But there are other three main concepts in training neural networks: bias, activation function, and loss function.

**Bias:** In a neural network, bias is like an extra number added to the neuron's calculation. It helps the neuron make better decisions by shifting the output, like adjusting the starting point of a line on a graph. Without bias, a neuron might be too limited in what it can learn. Bias gives more flexibility to the neuron, helping it fit the data better and make accurate predictions.

**Activation Function:** After a neuron processes input by multiplying it with its weight and adding bias, the activation function is applied. This function decides whether the neuron should pass its information forward. It also gives the neuron non-linear properties, which means the network can handle complex data. Without an activation function, the network would only work with simple, linear data and wouldn't perform well on real-world, complex data.

**Loss Function:** The loss function measures how accurate the network's predictions are. A smaller loss value means the difference between the actual data and the predicted data is less, so the model is predicting more accurately. The main goal during training is to reduce the loss function, while everything else happens in the background.

Now, it's time to train a neural network and analyze a few key things: how the loss function decreases, and how the gradient (derivative) plays a role in training.

First, a `Value` class is created that defines all the methods needed during training, such as multiplication, addition, subtraction, division, power functions, and exponential functions. An activation function, `tanh`, is also defined. Lastly, backward functions are set up so that gradients in backpropagation are computed automatically. Below is a code snippet, and the complete code can be found in the link to the Colab notebook.

```python
class Value:
    def __init__(self, data, _children=(), _op='', lable=''):
        ...

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
```

The backward functions computes the gradient of two points which are connected with each other via `+` operator. Here the gradient just passes from the previous node to the next node, without any change. That's why `self.grad` is computed by only `out.grad` which is the previous node gradient. Now we are gonna create another `Neuron` class, which is the building block of a whole neural network. The code snippet is given below.

```python
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()  # Apply tanh activation
        return out

    def parameters(self):
        return self.w + [self.b]
```

The neuron is initialized with a list of random weights and bias between -1 and +1. The `act` function calculates the weighted sum inputs (`x`) and then add bias to it. It makes a linear equation which is then passed through `tanh` activation function. Similarly, we define another class `Layer` which represent layer of neurons. The layer class is a collection of neurons. Each neuron object is created within the layer class. The third class which is created is called `MLP`: a collection of layers. Each layer object is created within the `MLP` class. Complete code is provided in the notebook link at the end.

```python
xs = [
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
```

`xs` represents the input data to our neural network. There are four set of inputs each contains three values. `ys` are actual values we want our model to predict. `ypred` are those values which the model predicts. Initially, there is a difference between the values we want our model to predict and what the model predict. In simple words this difference is called loss. If the difference between the actual values and predicted values is larger, it means the loss is high. And the model is not predicting accurately. Conversely, if this difference is small, loss is small, and model is predicting accurately. Our ultimate goal is to minimize the loss and make it zero, means the actual and predicted values are same and model is trained with 100 percent accuracy.

The concept behind decreasing the loss in a neural network is simple. It involves understanding the gradient, which is the derivative used in backpropagation. The gradient can be thought of as a slope: a negative gradient indicates a downward slope, while a positive gradient indicates an upward slope.

When the gradient for a specific neuron is negative, it means that the neuron is on a downward slope. To reduce the loss, the neuron needs to keep moving in the same downward direction until it reaches the lowest point, where the loss is minimized (ideally zero). So, when the gradient is negative, the weight of that neuron is increased by adding a positive value. This moves the neuron further down the slope. On the other hand, if the gradient is positive, the neuron is on an upward slope. To reduce the loss, the neuron needs to move in the opposite direction, back down the slope, towards the bottom. This is done by decreasing the weight of that neuron, which helps to lower the loss.

In essence, the sign of the gradient (simple derivative) tells us whether to increase or decrease the weight of a neuron. By following this simple rule, the network gradually adjusts its weights and improves its accuracy over time. This is how the network learns to make better predictions and reduce the loss to near zero.

From this point onward, training will be divided into two parts, focusing on how a small bug can cause a neural network to behave abnormally and produce incorrect output. This bug is related to not resetting a gradient to zero after each iteration which can lead to a math range error in python. If this reset doesn't happen after each iteration (one forward pass and one backward pass), the gradient from each iteration starts accumulating. As a result, the gradient value becomes larger and larger over time. This leads to a situation where gradient no longer reflects the correct values needed for proper weight adjustment, to minimize the loss. Below are the results when gradient is not reset.

```python
tracking_dict = {}
for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
    # Backward pass
    loss.backward()
```

First we make a dictionary which will store the results of all iteration. There are total 20 iterations. And we begin forward pass, in which the neural network predict the output. And this loss is just a difference between actual and predicted output. And then there is a backward pass, where gradient has not been reset to zero.

```python
neuron = n.layers[1].neurons[0]
weight = neuron.w[0].data
gradient = neuron.w[0].grad
current_loss = loss.data
# Update
for p in n.parameters():
    p.data += -0.4 * p.grad
```

We've chosen a specific neuron for our analysis — the first neuron of the second layer. We're tracking how its gradient and weights change during each iteration to spot any abnormal behavior in the network.

The last step is the update. The value `-0.4` is the learning rate, which controls how much the weights are adjusted in the opposite direction of the gradient. The learning rate is always negative. This way, if the gradient is negative, the adjustment becomes positive and increases the weight. Similarly, if the gradient is positive, the adjustment becomes negative, decreasing the weight.

![Gradient and weight values across iterations without resetting gradient to zero](/blog/calculus-derivatives-in-neural-network-training/image_3.png)

If we look at the above results, the gradient is increasing after each iteration, and weights are changing in each iterations. But these changes aren't in any fixed pattern. Even though the step size is constant, and if we look at the loss after third iteration, from 3.4 it increased to 7.9. After it, gradients and weights are changing but the loss is constant for next few iteration. Again after 7.6 it got decreased to 1.1 and then after 12th iteration, python said, this gradient is very large and out of my range.

![Plot of gradient, weight, and loss without gradient reset — no fixed pattern](/blog/calculus-derivatives-in-neural-network-training/image_4.png)

The graph above shows the same results. The gradients and weights are not adjusting in a constant pattern. Gradient is negative, weight is increasing, but there seem no change in the loss.

Now, we will make a little adjustment, rest of the code will remain the same and magic will happen.

```python
# Backward pass
for p in n.parameters():
    p.grad = 0.0
loss.backward()
```

In the backward pass, the gradient will be reset to zero in every iteration, below are the results.

![Output of 30 iterations after resetting gradient to zero — clean training](/blog/calculus-derivatives-in-neural-network-training/image_5.png)

Above is the output of 30 iteration. There is no math range error, the gradient and weights are adjusting in a proper pattern, and loss is decreasing after every iteration. Which means that the difference between actual output and the one predicted by a model is decreasing. And eventually it gets approximately zero. Model has achieved its accuracy and it is properly trained. The graph below shows the same results.

![Plot showing loss steadily decreasing after gradient reset](/blog/calculus-derivatives-in-neural-network-training/image_6.png)

## Conclusion

In conclusion, training a neural networks involves several key steps, from setting up neuron with weight and biases to applying activation function and minimizing the loss. The process of backpropagation, which calculates gradients, is crucial in updating weights and improving network accuracy. However, it is essential to handle gradients carefully, particularly ensuring that they are reset after each iteration. Failure to do so can lead to incorrect weight adjustments and prevent the network from learning effectively.

[Open the full notebook on Colab →](https://colab.research.google.com/drive/173MQH-wbH-gVqVAaIjLwY4EuWKJYxmXH#scrollTo=qA1tXg_H_VR9)
