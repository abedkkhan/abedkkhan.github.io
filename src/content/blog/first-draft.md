---
title: "First Draft"
description: "On finetuning for creativity: the target you can only verify once, the proxy you optimize instead, and a reward curve that keeps climbing while the outputs collapse."
pubDate: 2026-09-04
readTime: "2 min read"
tags: ["rl", "reward-hacking", "creativity", "evaluation"]
---

![Cover image — a reward curve climbing while sampled outputs collapse into one shape](/blog/first-draft/cover.svg)

Some things can be checked. A proof either holds or it doesn't. A program compiles or it fails. You can build a reward around these and trust it, because the target sits still while you aim at it.

Creativity isn't like that. Neither is taste, or originality, or whatever we mean when we ask a model to produce something interesting. There's no test to run. And the strange part is that it isn't simply unverifiable — it's verifiable once. You can look at an output and say yes, that's fresh, that's unlike what came before. But the moment you've said it, the thing you were pointing at has moved. What was surprising becomes the reference point, and the bar for surprising sits somewhere else now. Verify it and you've spent it.

We finetune for these targets anyway. And to do that we need a number, so we build a proxy — a reward function, a scorer, some benchmark that correlates with the thing we can't state directly. Then we run RL against it and let the model climb.

The model never sees the target. It only ever sees the proxy. So: the reward goes up, and the outputs get worse.

Not worse in any way the numbers can see. On the benchmark everything is fine. Diversity scores hold. Semantic distance between generations looks healthy. But you read fifty samples and they're recognisably the same sample wearing different nouns. Somewhere in training the model found one shape that scored well, and it has been producing that shape ever since, more and more confidently.

The instinct is to blame the training. Wrong dataset, wrong hyperparameters, too many steps, not enough entropy at the end. So you tune. Raise the temperature, change the sampling, widen the data — and the model produces the same thing with more unusual word choices.

Eventually you notice the reward curve never stopped climbing. It went up the entire time the outputs were collapsing. That's the part worth sitting with. Nothing failed. The reward function did exactly what it was built to do, every step, and the result was a model that can't surprise you.
