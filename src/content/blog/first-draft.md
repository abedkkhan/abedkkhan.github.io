---
title: "First Draft"
description: "On finetuning for creativity: the target we can only verify once, the proxy we optimize instead, and a reward curve that keeps climbing while the outputs collapse."
pubDate: 2026-09-04
readTime: "4 min read"
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

## Deception

There is a paper, [Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://stars.library.ucf.edu/facultybib2010/1530/), and it names the failure precisely: deception. An objective encodes what the goal looks like and rewards resemblance to it. But resembling the goal and being on the route to it are different properties, and nothing makes them line up.

Their example is a Chinese finger trap. Pulling the fingers apart is the direct move and it tightens the trap. The move that works is pushing them together, which looks like the opposite of progress. Any score built on how close the fingers are to free will penalise the only action that frees them.

![A stylised architectural interior. Three figures stand on a cantilevered platform that juts out into empty air and goes no further, reached by a stair climbing from below. Behind them a far larger staircase rises away to the right, and the only route to its foot leads back down through the flights below.](/blog/first-draft/image_2.webp)

What makes this more than a puzzle is the generalisation: the objective function does not necessarily reward the stepping stones that lead to the objective. The intermediate states along the way often don't resemble the destination. Sometimes they resemble failure. And this is exactly what appears to be happening in a creativity finetune: an early draft of something genuinely unusual reads, to a reward model, as incoherent. Which is what it is. Coherence is what it acquires later.

So follow one of those intermediates through. It appears. It gets scored. It scores below what we already have, so selection discards it: that step, before it produces anything, before it can show what it was a step toward. Next round the same thing appears and is discarded again.

The route isn't something the search failed to find. It found it, repeatedly, and threw it out every time. And it threw it out correctly. Selection is supposed to remove lower-scoring candidates. That's the whole job. There's no malfunction anywhere in this, which is why tuning the training never helped: we were looking for a fault in a process that didn't have one.

Deception isn't a property of the search method. It's a property of what we chose to score, and it survives whatever we optimise it with. Their conclusion: while it seems natural to blame the search algorithm when search fails to reach the objective, the problem may ultimately lie in the pursuit of the objective itself.

## The Obvious Answer
