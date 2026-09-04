---
title: "First Draft"
description: "On finetuning for creativity: the target we can only verify once, the proxy we optimize instead, and a reward curve that keeps climbing while the outputs collapse."
pubDate: 2026-09-04
readTime: "2 min read"
tags: ["rl", "reward-hacking", "creativity", "evaluation"]
---

![Cover image: a nineteenth century academy hall, where dozens of students paint the same seated figure, beneath a wall of near identical accepted canvases from previous years](/blog/first-draft/cover.webp)

Some things can be checked. A proof either holds or it doesn't. A program compiles or it fails. We can build a reward around these and trust it, because the target sits still while we aim at it.

Creativity isn't like that. Neither is taste, or originality, or whatever we mean when we ask a model to produce something interesting. There's no test to run. And the strange part is that it isn't simply unverifiable. It's verifiable once. We can look at an output and say yes, that's fresh, that's unlike what came before. But the moment we've said it, the thing we were pointing at has moved. What was surprising becomes the reference point, and the bar for surprising sits somewhere else now. Verify it and we've spent it.

We finetune for these targets anyway. And to do that we need a number, so we build a proxy: a reward function, a scorer, some benchmark that correlates with the thing we can't state directly. Then we run RL against it and let the model climb.

The model never sees the target. It only ever sees the proxy. So: the reward goes up, and the outputs get worse.

Not worse in any way the numbers can see. On the benchmark everything is fine. Diversity scores hold. Semantic distance between generations looks healthy. But we read fifty samples and they're recognisably the same sample wearing different nouns. Somewhere in training the model found one shape that scored well, and it has been producing that shape ever since, more and more confidently.

The instinct is to blame the training. Wrong dataset, wrong hyperparameters, too many steps, not enough entropy at the end. So we tune. Raise the temperature, change the sampling, widen the data, and the model produces the same thing with more unusual word choices.

Eventually we notice the reward curve never stopped climbing. It went up the entire time the outputs were collapsing. That's the part worth sitting with. Nothing failed. The reward function did exactly what it was built to do, every step, and the result was a model that can't surprise us.
