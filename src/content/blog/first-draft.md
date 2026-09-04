---
title: "First Draft"
description: "On finetuning for creativity: the target we can only verify once, the proxy we optimize instead, and a reward curve that keeps climbing while the outputs collapse."
pubDate: 2026-09-04
readTime: "9 min read"
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

There is a paper by Joel Lehman and Kenneth Stanley, [Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://stars.library.ucf.edu/facultybib2010/1530/), and it names the failure precisely: deception. An objective encodes what the goal looks like and rewards resemblance to it. But resembling the goal and being on the route to it are different properties, and nothing makes them line up.

Their example is a Chinese finger trap. Pulling the fingers apart is the direct move and it tightens the trap. The move that works is pushing them together, which looks like the opposite of progress. Any score built on how close the fingers are to free will penalise the only action that frees them.

![A stylised architectural interior. Three figures stand on a cantilevered platform that juts out into empty air and goes no further, reached by a stair climbing from below. Behind them a far larger staircase rises away to the right, and the only route to its foot leads back down through the flights below.](/blog/first-draft/image_2.webp)

What makes this more than a puzzle is the generalisation: the objective function does not necessarily reward the stepping stones that lead to the objective. The intermediate states along the way often don't resemble the destination. Sometimes they resemble failure. And this is exactly what appears to be happening in a creativity finetune: an early draft of something genuinely unusual reads, to a reward model, as incoherent. Which is what it is. Coherence is what it acquires later.

So follow one of those intermediates through. It appears. It gets scored. It scores below what we already have, so selection discards it: that step, before it produces anything, before it can show what it was a step toward. Next round the same thing appears and is discarded again.

The route isn't something the search failed to find. It found it, repeatedly, and threw it out every time. And it threw it out correctly. Selection is supposed to remove lower-scoring candidates. That's the whole job. There's no malfunction anywhere in this, which is why tuning the training never helped: we were looking for a fault in a process that didn't have one.

Deception isn't a property of the search method. It's a property of what we chose to score, and it survives whatever we optimise it with. Their conclusion: while it seems natural to blame the search algorithm when search fails to reach the objective, the problem may ultimately lie in the pursuit of the objective itself.

## The Obvious Answer

If the objective is the problem, the obvious question is whether anything has ever searched without one.

Something has. Evolution ran for roughly four billion years with no goal specified anywhere. Nothing was aiming at eyes, or flight, or us. There was no fitness function in the sense we mean it. In biology, fitness isn't a target handed to an organism, it's a tally of who happened to leave descendants, counted afterward. It describes an outcome. It doesn't direct anything.

And that undirected process produced more genuine novelty than any designed search we've built. It couldn't get permanently stuck, because there was nothing to get stuck on: no peak that stayed optimal, no single measure everything was climbing. It just kept generating variation, and complexity accumulated as a byproduct.

There's an argument, Stephen Jay Gould's, for why complexity shows up at all when nothing is selecting for it. Life began about as simple as it is possible to be, because there was nowhere else to begin: nothing can be simpler than the simplest thing that copies itself. That's a hard floor, and there's no matching ceiling above. So undirected variation has only one direction with room in it, and complexity rises over time without anything pulling it there. [Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) cite this reading (a growing if contested one in biology, that the rise in complexity is a passive force rather than a selected one) and add a second mechanism to it. A search that rewards novelty burns through the simple options, because there are only so many ways to be simple. Once those are used up, the only way left to be new is to be more complex. Complexity as a forced move rather than a drift, arrived at by nothing but exhaustion.

[Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) make the same observation from the other side. They note a view in biology that the drive toward complexity isn't primarily driven by selection at all: that selection can actively suppress it, because if selection pressure is too high, any deviation from a locally optimal behaviour gets filtered out. The same mechanism, running too hard, removes the things that would have led somewhere.

Which gives a clean line: the objective is the trap, nature had no objective, so the escape route is to be more like evolution.

The line doesn't hold.

## Two Kinds of Evolution

The word is doing two jobs.

Copying evolution into code isn't new. Evolutionary algorithms have done it for decades, and the mechanics transfer cleanly: a population of candidates, random mutation, selection of the ones that do better, repeat. Genetic algorithms and evolution strategies are both that loop.

But something gets added in the translation, and it isn't small. Nature had no target. An evolutionary algorithm needs one, because selection has to be told what "better" means. So a fitness function goes in (written by someone, pointing at a goal) and from that moment the algorithm is maximising a number the way any other optimiser does. Population and mutation are the parts inherited from nature. The objective is the part that wasn't there.

Which means "be more like evolution" splits into two different instructions, and they are not the same size. Copying the search is easy. Dropping the goal is the hard one, because selection has to be told something.

Most of the field takes the easy half. [Salimans and coauthors](https://arxiv.org/abs/1703.03864) present evolution strategies as an alternative to Q-learning and policy gradients, not as an alternative to having an objective. Same returns, same environments, same number being maximised. The selling points are wall-clock time, parallelism, and tolerance for long horizons: every one a claim about how we move through the space, none about what we are moving toward.

[Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) take the hard half, and it is the point of their paper. Novelty search abandons the objective outright. Nothing is scored on how close it is to a goal. Candidates are kept for being unlike everything found so far, measured against an archive of the behaviour already seen, and in their experiments the maze gets solved by a population that was never told where the exit was. So the absence of a goal is not impossible to copy. It has been done.

What it doesn't do is remove the choice. Novelty search still needs someone to say what counts as different, and that description of behaviour is as much a human decision as a fitness function is. In a maze it's obvious: where the robot ended up. In a creativity finetune nothing is obvious. Whatever axis we pick, the model finds the cheapest way to be different along it. If that axis is embedding distance, the cheap answer is presumably to scatter. If it's vocabulary, the cheap answer starts to look like unusual word choices over the same shape, which is where we came in.

So an evolutionary algorithm on a deceptive objective is deceived. It has the population, it has the mutation, and it will use them to climb the same misleading landscape and settle on the same local optimum. It may see more of the space on the way there. That is worth something, and it isn't the thing we needed.

There's a natural objection: even if ES optimises the same objective, it searches differently, so perhaps it lands somewhere different. The answer is a qualified yes, and the qualification matters.

[Zhang, Clune and Stanley](https://arxiv.org/abs/1712.06564) found ES tracks gradient descent more closely than expected: its gradient estimate is poor, but poor turns out to be sufficient. [Lehman, Chen, Clune and Stanley](https://arxiv.org/abs/1712.06568) then showed this holds only when the search distribution is narrow. Widen it and ES starts optimising over a region of parameter space rather than a point, which gives it something the gradient doesn't have: it can cross a flat stretch with no signal in it, and it can pass over a narrow peak without being caught, because a wide enough search doesn't register a narrow trap as a trap.

![A reward landscape with a tall narrow spike on the left and a broad higher hill on the right. A narrow search sits squarely on the spike and settles there. A wide search spans the spike and the flat ground beyond it, and carries on toward the higher hill, because a peak that thin is only a sliver of a wide sample.](/blog/first-draft/image_3.svg)

That's a genuine escape from some deceptions. But it costs us others: the same work shows wide search failing where a narrow path has to be followed precisely. We're trading which traps catch us, not leaving the landscape that produces them. Deception is still a property of what we chose to score.

## One Change

Note what novelty search actually changed. Not the loop: same population, same mutation, same selection. Only the input to selection, from how close a candidate came to the goal to how unlike everything already seen it was.

Which settles what the result is evidence for. Not that evolutionary search is better, since it is the same search. The objective was doing the trapping. Remove the pull toward a goal and the traps stop being traps, not because we climb out of them but because nothing is holding us there.
