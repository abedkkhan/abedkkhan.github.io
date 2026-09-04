---
title: "What If the Objective Is the Trap"
description: "On finetuning for creativity: the target we can only verify once, the proxy we optimize instead, and a reward curve that keeps climbing while the outputs collapse."
pubDate: 2026-09-04
readTime: "16 min read"
tags: ["rl", "reward-hacking", "creativity", "evaluation"]
---

![Cover image: a nineteenth century academy hall, where dozens of students paint the same seated figure, beneath a wall of near identical accepted canvases from previous years](/blog/what-if-the-objective-is-the-trap/cover.webp)

Some things can be checked. A proof either holds or it doesn't. A program compiles or it fails. We can build a reward around these and trust it, because the target sits still while we aim at it.

Creativity isn't like that. Neither is taste, or originality, or whatever we mean when we ask a model to produce something interesting. There's no test to run. And the strange part is that it isn't simply unverifiable. It's verifiable once. We can look at an output and say yes, that's fresh, that's unlike what came before. But the moment we've said it, the thing we were pointing at has moved. What was surprising becomes the reference point, and the bar for surprising sits somewhere else now. Verify it and we've spent it.

We finetune for these targets anyway. And to do that we need a number, so we build a proxy: a reward function, a scorer, some benchmark that correlates with the thing we can't state directly. Then we run RL against it and let the model climb.

The model never sees the target. It only ever sees the proxy. So: the reward goes up, and the outputs get worse.

Not worse in any way the numbers can see. On the benchmark everything is fine. Diversity scores hold. Semantic distance between generations looks healthy. But we read fifty samples and they're recognisably the same sample wearing different nouns. It reads like the model found one shape that scored well somewhere in training, and has been producing that shape ever since, more and more confidently.

There's a reason that shape tends to be the safe one. Ethan Smith, in [The Mean Preference Is a Bad Estimate of Preferences](https://www.ethansmith2000.com/post/the-mean-preference-is-a-bad-estimate-of-preferences), argues that a reward model built by averaging ratings across people ends up pointing somewhere nobody actually stands. When tastes genuinely conflict, and they do, the average of them is the middle, and the middle is what nobody loves and nobody objects to. He treats that as a fixable defect and goes looking for better reward models. The other reading is that predicting the middle is simply the right answer to labels that disagree, which would make the blandness a description rather than a fault.

The instinct is to blame the training. Wrong dataset, wrong hyperparameters, too many steps, not enough entropy at the end. So we tune. Raise the temperature, change the sampling, widen the data, and the model produces the same thing with more unusual word choices.

Eventually we notice the reward curve never stopped climbing. It went up the entire time the outputs were collapsing. That's the part worth sitting with. Nothing failed. The reward function did exactly what it was built to do, every step, and the result was a model that can't surprise us.

## Deception

There is a paper by Joel Lehman and Kenneth Stanley, [Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://stars.library.ucf.edu/facultybib2010/1530/), and it names the failure precisely: deception. An objective encodes what the goal looks like and rewards resemblance to it. But resembling the goal and being on the route to it are different properties, and nothing guarantees they line up.

Their example is a Chinese finger trap. Pulling the fingers apart is the direct move and it gets nowhere. The move that works is pushing them together, which looks like the opposite of progress and feels like tightening the trap. Any score built on how close the fingers are to free will penalise the only action that frees them.

![A stylised architectural interior. Three figures stand on a cantilevered platform that juts out into empty air and goes no further, reached by a stair climbing from below. Behind them a far larger staircase rises away to the right, and the only route to its foot leads back down through the flights below.](/blog/what-if-the-objective-is-the-trap/image_2.webp)

What makes this more than a puzzle is the generalisation: the objective function does not necessarily reward the stepping stones that lead to the objective. The intermediate states along the way often don't resemble the destination. Sometimes they resemble failure. And a creativity finetune looks like the same shape of problem: an early draft of something genuinely unusual reads, to a reward model, as incoherent. Which is what it is. Coherence is what it acquires later.

So follow one of those intermediates through. It appears. It gets scored. It scores below what we already have, so selection discards it: that step, before it produces anything, before it can show what it was a step toward. Next round the same thing appears and is discarded again.

The route isn't something the search failed to find. It found it, repeatedly, and threw it out every time. And it threw it out correctly. Selection is supposed to remove lower-scoring candidates. That's the whole job. There's no malfunction anywhere in this, which would explain why tuning the training never helped: we were looking for a fault in a process that didn't have one.

Deception isn't a property of the search method. It's a property of what we chose to score, and it outlives most of what we throw at it. Their conclusion: while it seems natural to blame the search algorithm when search fails to reach the objective, the problem may ultimately lie in the pursuit of the objective itself.

## The Obvious Answer

If the objective is the problem, the obvious question is whether anything has ever searched without one.

Something has. Evolution ran for roughly four billion years with no goal specified anywhere. Nothing was aiming at eyes, or flight, or us. There was no fitness function in the sense we mean it. In biology, fitness isn't a target handed to an organism, it's a tally of who happened to leave descendants, counted afterward. It describes an outcome. It doesn't direct anything.

And that undirected process produced more genuine novelty than any designed search we've built. Whatever it got stuck on didn't stay a trap, because no peak stayed optimal and no single measure was there to climb. It just kept generating variation, and the ceiling on complexity kept lifting as a byproduct.

There's an argument, Stephen Jay Gould's, for why complexity shows up at all when nothing is selecting for it. Life began about as simple as it is possible to be, because there was nowhere else to begin: nothing can be simpler than the simplest thing that copies itself. That's a hard floor, and there's no matching ceiling above. So undirected variation has only one direction with room in it, and the upper bound on complexity rises over time without anything pulling it there. [Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) cite this reading (a growing if contested one in biology, that the rise in complexity is a passive force rather than a selected one) and add a second mechanism to it. A search that rewards novelty burns through the simple options, because there are only so many ways to be simple. Once those are used up, the only way left to be new is to be more complex. Complexity as a forced move rather than a drift, arrived at by nothing but exhaustion.

[Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) make the same observation from the other side. They note a view in biology that the drive toward complexity isn't primarily driven by selection at all: that selection can actively suppress it, because if selection pressure is too high, any deviation from a locally optimal behaviour gets filtered out. The same mechanism, running too hard, removes the things that would have led somewhere.

Which gives a clean line: the objective is the trap, nature had no objective, so the escape route is to be more like evolution.

The line doesn't hold.

## Two Kinds of Evolution

The word is doing two jobs.

Copying evolution into code isn't new. Evolutionary algorithms have done it for decades, and the mechanics transfer cleanly: a population of candidates, random mutation, selection of the ones that do better, repeat. Genetic algorithms and evolution strategies are both that loop.

But something gets added in the translation, and it isn't small. Nothing ever wrote nature's target down. Almost every evolutionary algorithm needs it in writing, because selection has to be told what "better" means. So a fitness function goes in (written by someone, pointing at a goal) and from that moment the algorithm is maximising a number the way any other optimiser does. Population and mutation are the parts inherited from nature. The objective is the part that wasn't there.

![A formal garden seen from the terrace above it, with open country beyond. In the foreground a parterre laid out to a drawing made before anything was planted, its beds answering each other in a pattern that only resolves when seen from up here. Past the boundary wall the same land continues as woodland that arranged itself, with no line in it drawn by anybody. Same soil, same rain, same seasons on both sides of the wall.](/blog/what-if-the-objective-is-the-trap/image_4.webp)

Which means "be more like evolution" splits into two different instructions, and they are not the same size. Copying the search is easy. Dropping the goal is the hard one, because nothing can be selected without a way of ranking it.

Much of what gets built takes the easy half. [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864), by Salimans and coauthors, presents evolution strategies as an alternative to Q-learning and policy gradients, not as an alternative to having an objective. Same returns, same environments, same number being maximised. The selling points are wall-clock time, parallelism, and tolerance for long horizons: every one a claim about how we move through the space, none about what we are moving toward.

[Lehman and Stanley](https://stars.library.ucf.edu/facultybib2010/1530/) take the hard half, and it is the point of their paper. Novelty search abandons the objective outright. Nothing is scored on how close it is to a goal. Candidates are kept for being unlike everything found so far, measured against an archive of the behaviours that were new when they appeared, and in their experiments the maze gets solved by a population that was never told where the exit was. So the absence of a goal is not impossible to copy. It has been done.

What it doesn't do is remove the choice. Novelty search still needs someone to say what counts as different, and that description of behaviour is as much a human decision as a fitness function is. In a maze it's obvious: where the robot ended up. In a creativity finetune nothing is obvious. Whatever axis we pick is an axis the model can be cheaply different along. If that axis is embedding distance, the cheap answer is presumably to scatter. If it's vocabulary, the cheap answer starts to look like unusual word choices over the same shape, which is where we came in.

Back to the ordinary case. An evolutionary algorithm on a deceptive objective is deceived. It has the population, it has the mutation, and it will use them to climb the same misleading landscape toward the same kind of local optimum. It may see more of the space on the way there. That is worth something, and it isn't the thing we needed.

## Better Machinery

There's a natural objection: even if evolution strategies optimise the same objective, they search differently, so perhaps they land somewhere different. The answer is a qualified yes, and the qualification matters.

Zhang, Clune and Stanley, in [On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent](https://arxiv.org/abs/1712.06564), found less difference than expected. Ordinary training reads the slope under its feet and steps uphill. Evolution strategies can only guess at that slope by sampling around themselves, and their guess is a poor one, but poor turns out to be good enough to go the same way. [ES Is More Than Just a Traditional Finite-Difference Approximator](https://arxiv.org/abs/1712.06568), by Lehman, Chen, Clune and Stanley, then showed this holds only while the sampling stays close in. Spread the samples wide and that resemblance breaks down. The method was always asking what is good across a whole neighbourhood rather than at a single point, and with enough spread it starts to show. That buys something the slope can never give it: it can cross flat ground where there is nothing to read, and it can walk over a narrow peak without noticing, because a peak that thin barely registers in a wide enough sample.

![A reward landscape with a tall narrow spike on the left and a broad higher hill on the right. A narrow search sits squarely on the spike and settles there. A wide search spans the spike and the flat ground beyond it, and carries on toward the higher hill, because a peak that thin is only a sliver of a wide sample.](/blog/what-if-the-objective-is-the-trap/image_3.svg)

That's a genuine escape from some deceptions. But it costs us others: the same work shows wide search failing where a narrow path has to be followed precisely. We're trading which traps catch us, not leaving the landscape that produces them. Deception is still a property of what we chose to score.

[Go-Explore](https://arxiv.org/abs/1901.10995), from Ecoffet and coauthors, is worth pausing on, because it cuts against all of this. It changes nothing about the reward and takes whatever the environment provides. What it changes is where an episode begins. The paper names two failures: detachment, where the reward that led to a promising place has already been spent, so nothing points back to it, and derailment, where it tries to explore on the way back and never arrives. The fix is an archive and a rule: return to a stored state deliberately, without exploring en route, and only then explore. The agent isn't smarter. It isn't allowed to forget, and it isn't allowed to wander on the way back.

That produced roughly four times the previous best on Montezuma's Revenge, an Atari game that became the benchmark for hard exploration. Pure mechanics, no objective touched. But notice what kind of hard Montezuma is. The reward is sparse and silent for long stretches, and it isn't lying. Nothing the game means to reward scores a bad state highly. The problem is reaching the states the reward already values, which is a different thing from a reward model that confidently prefers the bland completion. No amount of better exploring fixes the second. We would only be finding more efficient routes to the wrong place.

## One Change

Note what novelty search actually changed. Not the loop: same population, same mutation, same rule of keeping whatever scores highest. Only the score itself, from how close a candidate came to the goal to how unlike anything seen before it was.

Which narrows what the maze result is evidence for. Not that evolutionary search is better, since it is the same search. The objective was doing the trapping. Remove the pull toward a goal and the traps stop being traps, not because we climb out of them but because nothing is holding us there.

## Widening the Knob

So far the choice has looked binary: keep the objective or drop it. There is a third setting, and [POET](https://arxiv.org/abs/1901.01753), from Wang and coauthors, is where it shows up.

The domain is a two-legged robot learning to cross broken ground. Instead of fixing one obstacle course and training an agent against it, POET keeps generating new courses by mutating the old ones, trains an agent on each, and then periodically tries every agent on every other agent's course. If an agent does better somewhere other than where it was raised, it takes over there. That last step is the part that matters.

So there is no single problem being climbed. The reward function itself doesn't change, and it is the same one in every environment POET builds. What changes is the world it gets computed in. There's a growing population of problems, each with its own agent, and the whole thing keeps producing harder versions of itself. The paper's phrase for what these become is stepping stones, the same word Lehman and Stanley used eight years earlier, now describing solutions to intermediate problems that turn out to matter for problems generated later.

Run evolution strategies directly on one of the environments POET built and solved, and they get a small fraction of the score that counts as solving it. The failure is a familiar one. The agents learn to move forward, and then they learn to freeze in front of an obstacle, because falling costs a hundred points and standing still costs almost nothing. The reward stays positive the whole time. The behaviour has collapsed into one shape.

That transfer step is the mechanism worth taking. A solution that has plateaued on its own environment isn't discarded; it's tested somewhere else, where it might be exactly what's needed. The paper calls this goal switching, and it is how POET gets out of local optima. We escape the trap by changing which problem we're solving, not by climbing harder within it.

Which is a different escape from novelty search's. Novelty search refuses to repeat. POET makes repetition pointless, because the problem we succeeded at is no longer the only problem in front of us.

Three different places to intervene, then. Novelty search changes what selection reads. Go-Explore changes where the search begins and leaves the objective untouched. POET keeps the objective and multiplies the problems it gets applied to. The design space was never a switch.

## The Objection

There's a problem with the way we've described nature, and it's the one that survives everything above.

Ethan Smith made the point, in conversation, that this reading is too clean. Nature does have an objective: survive, reproduce. Nobody wrote it down, but it exists, and selection enforces it as ruthlessly as any fitness function. His framing: the objective in evolution is a function of competition and survival, and the conditions are constantly changing as the environment and the competitors for resources change.

That's the part we'd skipped. What nature lacks isn't an objective, it's a fixed one. Survival is defined against an environment and a set of rivals, and both move. Get faster and the predator gets faster. Find an unexploited niche and it fills. The bar isn't set anywhere. It's set relative to everything else, and everything else is adapting too.

So a local optimum in nature is on a clock. Whatever made something optimal was optimal against conditions that no longer hold. Nothing has to climb out of the trap. The trap dissolves underneath it.

Which is a different diagnosis from the one we've been running. Not "the objective is the problem" but "the fixed objective is the problem". And it points somewhere else: at self-play, at adversarial setups, at anything where the thing scoring us improves as we improve.

[Wang and coauthors](https://arxiv.org/abs/1901.01753) got there first, and they don't think self-play goes far enough. In most coevolutionary systems, they note, the part of the environment that isn't the opponent stays fixed. The rival moves and the world doesn't. No amount of coevolution against a fixed task, they argue, produces something that can write poetry or invent mathematics. The environment itself has to change, and eventually the reward function with it.

## What's Open

We don't have the answer.

The research here runs from 2011 to 2019. This isn't a survey of what's been built since, and it's likely parts of this have been addressed in ways worth knowing about. But the distinction survives regardless of what's been solved, because it isn't a claim about the state of the art. It's a claim about where an intervention sits, and much of what gets reached for when outputs collapse sits in the wrong place.

What might help, for the creativity case specifically, is something that treats the reward as a thing with a shelf life. Not a better reward model, but a reward model that expects to be wrong shortly and has some mechanism for noticing. POET moves the problem. Self-play moves the opponent. Neither is obviously the right shape for taste, which moves for reasons that have nothing to do with an adversary and everything to do with exposure: the thing gets seen, and seeing it is what wears it out.

Which points at something narrower than a better score, and it brings back the exhaustion argument from earlier. Lehman and Stanley used it about simple behaviours: there are only so many ways to be simple, so a search demanding novelty runs out of them and has to go somewhere more complex. The same shape seems to apply to taste, except what gets used up isn't simplicity, it's whatever everyone has already found. If a thing wears out by being seen, then part of what's worth rewarding is timing: being somewhere before it fills up. That's a property of the region, not of the output, and it isn't what a reward model is built to notice. Whether it could be trained into the weights rather than bolted on at sampling time is the part worth experimenting with.

If the target moves because measuring it moves it, then any fixed proxy is a photograph, and the question isn't how to take a better photograph. It's what we build instead.

Which leaves a fork underneath all of this. A target can fail to hold still for two different reasons. It moves because rivals adapt, which is nature's version, and it is the one self-play and POET are built for. Or it moves because it was looked at, and looking used it up. Those want different machinery. Taste might be either, or both at once in different proportions, and nothing above settles which.
