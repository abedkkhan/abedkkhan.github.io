---
title: "Visualizing the Activation, Gradients, Gradient to data ratio, and Update to Data Distributions in Neural Networks Training"
description: "A visual diagnostic tour through activation distributions, gradient flow, weight gradients, update-to-data ratios, and how batch normalization stabilizes them all."
pubDate: 2025-01-22
readTime: "18 min read"
tags: ["neural-networks", "batch-normalization", "gradients", "initialization", "pytorch"]
---

![Cover image — visualizing activations, gradients, and update-to-data ratios in neural network training](/blog/visualizing-activations-gradients-ratios/cover.webp)

In my last three blog posts, we explained about how to reduce the wrong but high confidence of neural network, how to identify dead neurons, and The Kaiming initialization and batch normalization. The core of these three blog post is, the correct initialization of neural network, more accurate training, and descent distribution of backward pass and forward pass activations and gradients.

1. [How to correctly Initialize the Neural Network: Mechanistic Interpretability Part 1](/blog/correctly-initialize-neural-network-mech-interp-1/)
2. [Identifying and Removing Dead Neurons in Training Neural Networks: Mechanistic Interpretability Part 2](/blog/dead-neurons-mech-interp-2/)
3. [The Kaiming initialization and Batch Normalization in Neural Networks](/blog/kaiming-init-batch-norm/)

I would highly recommend readers to go through these blog post in the above order before reading this blog post. Because, in this blog post we will inspect graphically, the changes we made to our neural network in the above blog posts, to make our neural network training and optimization more better and remove vanished gradients or dead neurons. Here, we will go directly into the initialization, so let's begin.

## Activation distribution

First we Visualize the the histogram of a forward pass activations at the `tanh` layers because they have a finite output `[-1, 1]` so its easy to visualize. To reduce the saturation or remove the dead neurons explained in blog post 2, we multiplied hidden layer weights with Kaiming initialization formula `gain / fan_in**0.5` at the initialization. There, we described that for different activation functions, we have different gain values such as for `tanh` we have `5/3`. Now, we use different gain values for `tanh` and see how activation distribution behave.

**Gain = 5/3**

```python
# gain 5/3
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
```

```
layer 1 (tanh): mean -0.02,   std 0.75,   saturated: 20.25%
layer 3 (tanh): mean -0.00,   std 0.69,   saturated: 8.38%
layer 5 (tanh): mean +0.00,   std 0.67,   saturated: 6.62%
layer 7 (tanh): mean -0.01,   std 0.66,   saturated: 5.47%
layer 9 (tanh): mean -0.02,   std 0.66,   saturated: 6.12%
```

![Activation distribution with gain 5/3 — stabilizes around std 0.66 across deeper layers](/blog/visualizing-activations-gradients-ratios/image_1.png)

We calculate the mean, standard deviation (std), and percent saturation of `t`. Percent saturation is defined as the proportion of values in `t` where `abs(t) > 0.97`. This basically tells us how many values are hitting the tails of the `tanh` function. When values hit the tails, the gradients become almost zero, effectively stopping learning. So, we don't want the percent saturation to be too high.

Looking at the graph and the stats, the first layer is pretty saturated, with a decent number of values hitting the tails. But as we go deeper into the network, the activations stabilize. The std settles around `0.66`, and the saturation drops to about `6%`. This stabilization shows that the network is handling activations better as we go deeper. It's a good sign because it means adding more layers can still work without the activations blowing up or dying.

The key reason for this stabilization and the smooth distribution of activations is the gain value of `5/3`. It balances the activations nicely across the layers, preventing them from exploding or collapsing. This makes the network trainable even with deeper architectures.

**Gain = 1**

```
layer 1 (tanh): mean -0.02, std 0.62, saturated: 3.50%
layer 3 (tanh): mean -0.00, std 0.48, saturated: 0.03%
layer 5 (tanh): mean +0.00, std 0.41, saturated: 0.06%
layer 7 (tanh): mean +0.00, std 0.35, saturated: 0.00%
layer 9 (tanh): mean -0.02, std 0.32, saturated: 0.00%
```

![Activation distribution with gain 1 — std shrinks toward zero as depth increases](/blog/visualizing-activations-gradients-ratios/image_2.png)

When we set the gain to `1` (or effectively no gain), we can see that the standard deviation (std) keeps shrinking as we go deeper into the layers, and the percent saturation basically drops to zero. The first layer looks decent, but after that, the activations just keep getting smaller and smaller, slowly heading towards zero.

It's because of the `tanh` layers. If we had just a stack of linear layers, initializing weights the way we've done before would've preserved the std at around `1` throughout the layers. But here, the `tanh` layers are squashing the activations. That's what `tanh` does — it takes input and compresses it into the range `[-1, 1]`. Every time a `tanh` layer squashes the activations, the std gets smaller.

This is why using some gain is important. The gain helps counteract this squashing effect by stretching out the activations a little. Without it, as we see here with `gain = 1`, the activations just keep shrinking deeper into the network. And when the activations shrink like this, the gradients also become tiny, which makes learning harder. If the gain is too small, like `1`, the activations will just collapse towards zero, and the network won't train well.

**Gain = 3**

```
layer 1 (tanh): mean -0.03, std 0.85, saturated: 47.66%
layer 3 (tanh): mean +0.00, std 0.84, saturated: 40.47%
layer 5 (tanh): mean -0.01, std 0.84, saturated: 42.38%
layer 7 (tanh): mean -0.01, std 0.84, saturated: 42.00%
layer 9 (tanh): mean -0.03, std 0.84, saturated: 42.41%
```

![Activation distribution with gain 3 — heavy saturation near ±1 across all layers](/blog/visualizing-activations-gradients-ratios/image_3.png)

Now, with the gain set to `3`, we see that the standard deviation (std) is consistently around `0.84` across all layers, and the percent saturation shoots up to nearly `40–47%`. What does that mean? The activations are hitting the tails of the `tanh` function way more often, meaning a lot of the values are squashed to either `-1` or `1`.

Why is this happening? It's because the gain is too high. When we initialize with a high gain, the weights get scaled up significantly, causing the activations to spread out more. But when the activations get too big, the `tanh` function starts flattening them out at its limits. That's the "saturation" we're seeing here — almost half the activations are stuck at the extremes (`-1` or `1`). This is bad because, at the tails, the gradient becomes almost zero, which kills learning. If the gradients are dead, our model won't update properly during training.

**Gain = 0.5**

```
layer 1 (tanh): mean -0.01, std 0.41, saturated: 0.00%
layer 3 (tanh): mean -0.00, std 0.20, saturated: 0.00%
layer 5 (tanh): mean +0.00, std 0.10, saturated: 0.00%
layer 7 (tanh): mean +0.00, std 0.05, saturated: 0.00%
layer 9 (tanh): mean -0.00, std 0.02, saturated: 0.00%
```

![Activation distribution with gain 0.5 — activations collapse toward zero in deeper layers](/blog/visualizing-activations-gradients-ratios/image_4.png)

Now, with the gain set to `0.5` (like super low), the activations are basically collapsing as we go deeper into the layers. Look at the std — it starts at `0.41` in the first layer but keeps shrinking down to just `0.02` in the last layer. The saturation is zero everywhere, meaning none of the activations are hitting the tails of `tanh` (no values close to `-1` or `1`). This makes sense because the activations are barely spreading out.

When the gain is super low, the weights are initialized with very small values. That means the outputs of each layer are tiny right from the start. Then, the `tanh` layers squash these small values even more, compressing them closer to zero. This keeps happening layer after layer, so the activations keep shrinking to the point where they're basically useless. It's like the signal dies out as it moves deeper into the network.

It turns out that a gain of `5/3` is a good choice for a network with linear layers and `tanh` activations. It keeps the std stable at a reasonable level across layers, preventing this shrinkage problem. But when the gain is too small or too high everything just collapse and vanish into nothingness. And we get dead neurons in our neural network.

## Gradient distribution

**Gain = 5/3**

```
layer 1 (tanh): mean +0.000010, std 4.205588e-04
layer 3 (tanh): mean -0.000003, std 3.991179e-04
layer 5 (tanh): mean +0.000003, std 3.743020e-04
layer 7 (tanh): mean +0.000015, std 3.290473e-04
layer 9 (tanh): mean -0.000014, std 3.054035e-04
```

![Gradient distribution with gain 5/3 — gradients remain well-scaled across layers](/blog/visualizing-activations-gradients-ratios/image_5.png)

So, now we're looking at the gradient distribution when the gain is set to `5/3`. First thing to notice — the gradients look pretty decent. The mean is hovering close to zero (which is what we want), and the standard deviation (std) gradually decreases as we go deeper into the layers. For example, the first layer has a std of around `4.2e-4`, and by the last layer, it's down to `3.0e-4`. It's shrinking, but not collapsing to zero or blowing up. That's a good sign.

Now, why are we getting this? This goes back to the activation distribution we saw earlier. With the gain of `5/3`, the activations were well-balanced — not too small and not too saturated. That's key because the gradients are directly tied to the activations. If the activations are too small (like with a low gain), the gradients would vanish. If the activations were too large or saturated (like with a high gain), the gradients would explode. But here, with `5/3`, the activations had a stable std (around `0.66`), which gives us nicely scaled gradients that don't vanish or explode.

This balance is why the gradients look so clean here. The gain of `5/3` prevents the `tanh` layers from squashing the activations too much, and it keeps the signal flowing through the network. That means the gradients can propagate backward effectively without breaking down. It's like a sweet spot where the network stays trainable and gradients stay healthy.

**Gain = 3**

```
layer 1 (tanh): mean -0.000001, std 9.977493e-04
layer 3 (tanh): mean +0.000010, std 7.421208e-04
layer 5 (tanh): mean +0.000003, std 5.569782e-04
layer 7 (tanh): mean +0.000017, std 3.952166e-04
layer 9 (tanh): mean -0.000014, std 3.051525e-04
```

![Gradient distribution with gain 3 — sharp drop-off in std across deeper layers](/blog/visualizing-activations-gradients-ratios/image_6.png)

With the gain set to `3`, the gradient distribution starts to change in a way that's connected to the activation saturation we saw earlier. Look at the std of the gradients — it starts out pretty big at the first layer (`9.97e-4`), but as you go deeper, it keeps shrinking until the last layer, where it's down to around `3.05e-4`. That's a pretty sharp drop-off.

Why is this happening? The gain of `3` is too high, which caused the activations to saturate earlier, as we saw in the activation distribution. Remember, when the `tanh` saturates (values near `-1` or `1`), the gradient basically gets squashed to near-zero for those activations. This means fewer effective gradients are flowing back through the network. So, while the first layer still has a decent std for the gradients, by the time we get to the deeper layers, most of the signal has been lost because the activations are heavily saturated.

This behavior is tied directly to the gain being too large. A high gain makes the initial weights bigger, which causes the activations to spread too far, hitting the tails of `tanh`. Once that happens, the gradient information starts dying out deeper into the network. The network's learning capability suffers because those small gradients mean smaller weight updates, making it harder to optimize the deeper layers.

**Gain = 0.5**

```
layer 1 (tanh): mean +0.000000, std 1.892402e-05
layer 3 (tanh): mean -0.000001, std 3.943546e-05
layer 5 (tanh): mean +0.000004, std 8.035369e-05
layer 7 (tanh): mean +0.000009, std 1.561152e-04
layer 9 (tanh): mean -0.000014, std 3.053498e-04
```

![Gradient distribution with gain 0.5 — early layers nearly dead, only deeper ones slowly come alive](/blog/visualizing-activations-gradients-ratios/image_7.png)

With the gain set to `0.5`, the gradient distribution is almost completely squashed for the earlier layers. Check out the std values — they're tiny at the start (`1.89e-5` for layer 1) and gradually grow as we go deeper, but even by the last layer, they're still small compared to what we saw with a proper gain like `5/3`. What this means is that the gradients are nearly dead in the earlier layers and only start to come alive a little in the deeper ones.

The super low gain makes the weights really small, which already gives us tiny activations. Then, when the `tanh` layers squash those tiny activations even further, the gradients shrink even more because the derivative of `tanh` is very small for values near zero. It's like the signal dies right at the start and has almost nothing left to propagate back through the network.

Notice how the std values in the gradients slowly increase as we go deeper into the network. That's because the deeper layers are slightly less affected by the earlier squashing, but it's still not enough to make the gradients meaningful. The network can't learn effectively because the gradient updates are so small that the weights barely change. This is why we need a higher gain like `5/3` — it balances the activations and keeps the gradients in a usable range. Without it, the network is basically paralyzed for learning.

## No tanh

**Gain 5/3 — Forward Pass**

```
layer 0 (Linear): mean -0.04, std 1.65, saturated: 55.12%
layer 1 (Linear): mean -0.04, std 2.72, saturated: 71.78%
layer 2 (Linear): mean -0.00, std 4.67, saturated: 81.31%
layer 3 (Linear): mean +0.09, std 7.50, saturated: 91.19%
layer 4 (Linear): mean -0.72, std 12.78, saturated: 93.69%
```

![Forward-pass activations without tanh — activations explode deeper in the network](/blog/visualizing-activations-gradients-ratios/image_8.png)

**Gain 5/3 — Backward Pass**

```
layer 0 (Linear): mean +0.000053, std 2.619185e-03
layer 1 (Linear): mean -0.000006, std 1.583188e-03
layer 2 (Linear): mean +0.000043, std 9.519162e-04
layer 3 (Linear): mean +0.000019, std 5.457934e-04
layer 4 (Linear): mean -0.000013, std 3.161244e-04
```

![Backward-pass gradients without tanh — gradients shrink deeper in the network](/blog/visualizing-activations-gradients-ratios/image_9.png)

Now we remove all the `tanh` layers, keep the gain at `5/3`, and turn the network into a giant linear sandwich. Let's break down what happens to the forward-pass activations and backward-pass gradients.

**Forward-pass Activations:**

In the activation graph, things start out with the blue line (layer 0), which looks fairly reasonable. But by the time we get to layer 4, the activations have completely blown up — they've become super diffused. Look at the stats: the std starts at `1.65` in layer 0 and keeps exploding, ending up at `12.78` in layer 4. The percent saturation also skyrockets, hitting `93.69%` in the last layer, which means most of the values are in the extremes (large positive or negative). This happens because there's nothing in the network (like `tanh`) to squash or stabilize the activations. With just linear layers, the activations grow out of control.

**Backward-pass Gradients:**

Now look at the gradient graph. The stats show the opposite problem. Gradients start relatively large in the early layers (layer 0 has a std of `2.61e-3`), but as we go deeper, they shrink significantly. By the time we reach layer 4, the std is down to `3.16e-4`. This means the deeper layers get less effective gradient updates, and learning there slows down. The gradient distribution shows this clearly — the deeper layers are barely getting any meaningful gradients.

This setup creates an asymmetry in the network. On the forward pass, activations are exploding. On the backward pass, gradients are vanishing. It's like the two sides of the network are fighting each other. When we use a gain of `5/3` with just linear layers, the weights are too large, and there's nothing to stabilize the activations or gradients.

If we flip it and use a very small gain (like `0.5`), the opposite would happen: activations would shrink (collapse to zero), and gradients would start diffusing, making them useless for training.

Without `tanh` (or any nonlinear activations), the gain has to be carefully tuned — specifically, it should be closer to `1` to keep both the activations and gradients in a reasonable range. This shows how fragile training was before modern techniques like batch normalization (discussed below), advanced optimizers like Adam, or residual connections. Back in the day, training a neural network was like balancing a pencil on your finger. Everything — activations, gradients, weights — had to be precisely orchestrated, or the whole thing would fall apart.

## Weights gradients distribution

Now we're adding another important plot here. Ultimately, when we train a neural net, we're updating its parameters. So, it's crucial to analyze the parameters, their values, and their gradients. For simplicity, we're keeping this to 2D weights and skipping bias terms, gammas, betas, and batch normalization.

We introduce the gradient-to-data ratio here. This gives us an idea of how the scale of the gradients compares to the scale of the actual parameter values (the weights). This is important because during training, the step update we apply is the learning rate times the gradient. If the gradient values are too large compared to the parameter values, things can go wrong — our updates might overshoot, destabilizing training.

```python
# visualize histograms of parameters
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');
```

```
weight   (27, 10) | mean -0.000031 | std 1.365078e-03 | grad:data ratio 1.364090e-03
weight  (30, 100) | mean -0.000049 | std 1.207430e-03 | grad:data ratio 3.871660e-03
weight (100, 100) | mean +0.000016 | std 1.096730e-03 | grad:data ratio 6.601988e-03
weight (100, 100) | mean -0.000010 | std 9.893572e-04 | grad:data ratio 5.893091e-03
weight (100, 100) | mean -0.000011 | std 8.623432e-04 | grad:data ratio 5.158124e-03
weight (100, 100) | mean -0.000004 | std 7.388576e-04 | grad:data ratio 4.415211e-03
weight  (100, 27) | mean -0.000000 | std 2.364824e-02 | grad:data ratio 2.328203e+00
```

![Weights gradient distribution at initialization — last layer dominates with outsized gradients](/blog/visualizing-activations-gradients-ratios/image_10.png)

In above example, most of the gradient-to-data ratios are pretty small. The gradients are roughly `1000` times smaller than the weight values for most layers, which is good — it means our updates are scaled nicely relative to the data. But here's the catch: the last layer (pink line) is the troublemaker.

The last layer's gradient std is about `10x` larger than the gradients in the rest of the network. This means the last layer is being updated much faster compared to the other layers. At initialization, this creates an imbalance where the last layer is effectively learning `10` times faster than the rest of the network.

This imbalance can be problematic in a simple stochastic gradient descent (SGD) setup. If the last layer updates too fast, it can dominate the training process, making it harder for the other layers to catch up and learn effectively. This issue is specific to the way the network is initialized, where the last layer's gradients are out of proportion with the rest.

If we train for a while (e.g., include a condition like `if i > 1000, break`), this imbalance starts to fix itself as shown in the example below. Over time, the last layer (pink tail) shrinks, and its gradient values become more in line with the rest of the network. This naturally balances out during optimization.

```
weight   (27, 10) | mean +0.000171 | std 1.366265e-02 | grad:data ratio 1.361588e-02
weight  (30, 100) | mean -0.000142 | std 1.063916e-02 | grad:data ratio 3.376994e-02
weight (100, 100) | mean -0.000085 | std 8.917154e-03 | grad:data ratio 5.224393e-02
weight (100, 100) | mean -0.000048 | std 7.586985e-03 | grad:data ratio 4.421483e-02
weight (100, 100) | mean +0.000053 | std 7.078861e-03 | grad:data ratio 4.147504e-02
weight (100, 100) | mean +0.000024 | std 6.234779e-03 | grad:data ratio 3.687952e-02
weight  (100, 27) | mean -0.000000 | std 2.015578e-02 | grad:data ratio 2.435712e-01
```

![Weights gradient distribution after training — imbalance smooths out, last layer less extreme](/blog/visualizing-activations-gradients-ratios/image_11.png)

Looking at the numbers, the gradient-to-data ratios have improved across the board. Most layers now have ratios around `0.01` to `0.05`, which means the gradients are still much smaller compared to the parameter values. This is a good range — it keeps the updates stable and avoids any large, destabilizing jumps in parameter values.

From the graph, we can see that the gradients are starting to balance out across the layers. The peaks for all layers are closer together, and the tail of the pink line (last layer) is no longer as extreme as before. This shows that the longer training time helped smooth out the imbalance to some extent, but the last layer still stands out.

The gradient-to-data ratio is a critical metric to monitor during initialization. While it looks good for most layers here, the last layer stands out as an outlier with larger gradients. If left unchecked with simple SGD, this can lead to uneven training dynamics. Modern optimizers like Adam or batch normalization are much better at handling this imbalance and would smooth out these issues during training.

## Update-to-data ratio

Now, let's look at the last important plot: the update-to-data ratio. While the gradient-to-data ratio is useful, it doesn't tell the full story. What really matters in training is the update-to-data ratio because it reflects how much we're actually changing the parameters in each iteration.

At each iteration, the update is calculated as the learning rate times the gradient. This update is then applied to the parameter values. We track the std of the updates divided by the std of the parameter values (the actual data). This ratio tells us how significant the updates are compared to the parameter values.

```python
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);
```

![Update-to-data ratio across training — most layers hover near the ideal 1e-3 guideline](/blog/visualizing-activations-gradients-ratios/image_12.png)

In the graph, we can see how the update-to-data ratios change over the course of training. At initialization, the ratios start at certain values, and during training, they stabilize. We plot an approximate guide for what the ratio should ideally be, using: `plt.plot([0, len(ud)], [-3, -3], 'k')`. This corresponds to `~1e-3`. It means the updates should ideally be about `1/1000th` of the parameter values. For example, in param 0 (blue line), the ratio looks good and stabilizes around the guide line.

The final layer (pink line) is an outlier because it was artificially scaled down during initialization to prevent the softmax layer from being too confident. This makes the parameter values in this layer smaller, so the update-to-data ratio temporarily spikes. Over time, this stabilizes as the weights in this layer start to learn.

If the learning rate is too low (e.g., `0.001`), the updates will be tiny, and the ratios will drop way below `1e-3`. This means the parameters are barely learning and training is too slow as shown below.

- **Removing `fan_in ** 0.5` in Initialization**

    If we remove the scaling factor (`fan_in ** 0.5`) during initialization, the whole training process breaks:

    - **Activation Saturation:** The activations saturate early, meaning many values hit the extremes.
    - **Messed-Up Gradients:** The gradients become imbalanced and unreliable.
    - **Broken Ratios:** Both the gradient-to-data and update-to-data ratios become chaotic.
    - **Update-to-Data Disaster:** In the below graph, removing proper initialization would show layers learning at wildly different speeds, with some layers updating way too fast.

![Update-to-data ratio after removing fan_in scaling — chaotic, layers learn at wildly different speeds](/blog/visualizing-activations-gradients-ratios/image_13.png)

This plot is a critical diagnostic tool for understanding if your learning rate and initialization are working well. A good learning rate keeps the update-to-data ratio near `~1e-3` for most layers. Here, the setup looks decent, but the learning rate might be slightly high. Overall, everything stabilizes well, and the last layer resolves its imbalance over time.

## Batch Normalization

Now we introduce batch normalization, and this changes how the network handles activations, gradients, and updates. With batch norm, we don't have to stress about setting the right gain anymore. Batch norm sits before the non-linearity and normalizes the outputs of each layer during training. Here's what happens:

- **Last Layer Handling:**

    - Previously, we manually scaled down the last layer's weights to make the softmax less confident.
    - With batch norm, we don't touch the weights. Instead, the gamma parameter (`bnmean`) of the batch norm takes care of this. Gamma multiplicatively adjusts the normalized values, so we can rely on it to control the output confidence.

- **Activation Distribution**

    ```
    layer 2  (tanh): mean -0.00, std 0.63, saturated: 2.62%
    layer 5  (tanh): mean +0.00, std 0.64, saturated: 2.47%
    layer 8  (tanh): mean -0.00, std 0.64, saturated: 2.16%
    layer 11 (tanh): mean +0.00, std 0.65, saturated: 1.81%
    layer 14 (tanh): mean -0.00, std 0.65, saturated: 1.78%
    ```

    ![Activation distribution with batch norm — std stable at ~0.65, saturation low across all layers](/blog/visualizing-activations-gradients-ratios/image_14.png)

    Activations look fantastic. Every `tanh` layer is normalized by batch norm, so the standard deviation is stable at `~0.65` throughout all layers. Saturation is low (around `2%`), and the activations stay consistent across the network. This shows how batch norm eliminates the problem of exploding or vanishing activations.

- **Gradient Distribution**

    ```
    layer 2  (tanh): mean -0.000000, std 3.682961e-03
    layer 5  (tanh): mean -0.000000, std 3.262612e-03
    layer 8  (tanh): mean +0.000000, std 2.973734e-03
    layer 11 (tanh): mean +0.000000, std 2.741114e-03
    layer 14 (tanh): mean +0.000000, std 2.584295e-03
    ```

    ![Gradient distribution with batch norm — smooth and well-behaved, no vanishing or exploding](/blog/visualizing-activations-gradients-ratios/image_15.png)

    Gradients are smooth and well-behaved across the layers. No vanishing or exploding gradients here, thanks to the normalization applied in batch norm.

- **Weight Gradient Distribution**

    ![Weight gradient distribution with batch norm — balanced across layers, no wild variation](/blog/visualizing-activations-gradients-ratios/image_16.png)

    The weight gradients also look good. There's no wild variation or imbalance between layers.

- **Update-to-Data Ratios**

    ![Update-to-data ratios with batch norm — all parameters update at similar, stable rates](/blog/visualizing-activations-gradients-ratios/image_17.png)

    Updates look reasonable. Most parameters are updating at the same rate, and the ratios hover close to the ideal range. While they are slightly above `-3`, they're still stable and not too far off.

With batch norm, the network becomes less brittle to the initialization gain. For example: If we lower the gain to `0.2` (way less than `5/3`), the activations, gradients, and weight gradients remain fine because batch norm explicitly normalizes them.

![Update-to-data ratio with batch norm and gain 0.2 — activations stable, but ratio shifts due to batch-norm backward pass](/blog/visualizing-activations-gradients-ratios/image_18.png)

However, the update-to-data ratio changes as shown above. This happens because the backward pass of batch norm adjusts how the activations interact with the parameter updates, slightly altering the scale of the updates.

If we increase the learning rate by `10x` (from `0.1` to `1.0`), the updates or the blue line, shift above the black line which is closer to the ideal setting.

So, with batch norm, the network becomes much more stable, and issues like vanishing/exploding activations and gradients are handled. The reliance on precise initialization (e.g., `fan_in**0.5`, gain) is greatly reduced. The update-to-data ratio becomes the critical metric to monitor, as it tells us whether the learning rate needs adjustment.

## Conclusion

In this exploration, we started by analyzing the intricate dynamics of activations, gradients, and parameter updates in neural networks, emphasizing how initialization, gain, and network architecture impact training. We observed that without proper tuning, activations can either explode or vanish, gradients can shrink or become unstable, and the updates to parameters can skew training dynamics. Using techniques like setting the gain to `5/3` for `tanh` layers provided stability, but we saw how networks remain sensitive to initialization, especially in deeper architectures. Introducing batch normalization was a game-changer — it normalized activations across layers, making the network less brittle to initialization and gain, while ensuring consistent gradient flow and parameter updates. Batch norm also allowed us to focus on update-to-data ratios, helping us fine-tune the learning rate effectively. Overall, we've seen that training a neural network involves balancing activations, gradients, and updates, and modern techniques like batch norm significantly simplify this process, making deep networks more robust and easier to train.

[GitHub code →](https://github.com/abedkkhan/ML-topics-I-love-exploring/blob/main/Visualizing%20the%20Activation%2C%20Gradients%2C%20Gradient%20to%20data%20ratio%2C%20and%20Update%20to%20Data%20Distributions%20in%20Neural%20Networks%20Training.py)
