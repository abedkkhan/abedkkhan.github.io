---
title: "LLMs Will Always Generate Plausible yet Incorrect Output and We Have to Make Peace With it: A Paper Review"
description: "A personal take on 'LLMs Will Always Hallucinate, and We Need to Live With This' — the paper argues hallucination is an inherent, mathematically inevitable feature of LLMs, not just an occasional error."
pubDate: 2024-10-04
readTime: "12 min read"
tags: ["llms", "hallucination", "paper-review", "godel", "turing-machines"]
---

![Cover image — LLMs will always hallucinate paper review](/blog/llms-always-hallucinate-paper-review/cover.webp)

Large Language models have had a tremendous growth across all domains since past few years. Researchers are actively engaged in finetuning these models, increasing parameters, context and token length, as well as developing new architectures for their better performance. But, unfortunately, as we make advancements in LLMs, we also come across various issues and limitations of these large language models. One of the biggest limitation of LLMs is generating plausible yet incorrect outputs: **Hallucination**. It means when a language model gives an output but that output is not 100% based on facts, not 100% correct, as well as not even 100% aligned with its trained data and retrieved information which is also called `RAG`. Over the course of time, various techniques have been applied, but hallucination isn't coped up with completely. And the paper I am gonna cover which is: *"LLMs Will Always Hallucinate, and We Need to Live With This"* makes some interesting claims and prove them mathematically.

The paper states that hallucination in language models are not just occasional errors but an inherited feature of these models. It also explains that hallucination originates from the fundamental mathematical and logical structure of LLMs. It is, therefore, impossible to eliminate hallucination through improving architecture, datasets enhancements, or fact checking mechanisms. The paper draws its analysis on one of the remarkable theorem of mathematics which is Gödel's First Incompleteness Theorem. It explains the undecidability of a problem like halting, emptiness, and acceptance problem. Will demonstrate this later in this blog.

First, the paper explains the fundamental concepts of how LLMs generate an output based on conditional probability which means predict the linguistic patterns and the conditional probability of next token given the previous token.

![Conditional probability formulation for next-token prediction in LLMs](/blog/llms-always-hallucinate-paper-review/image_1.jpeg)

The paper also used various improvements strategies to improve the performance of LLMs such as rotatory positional encodings in the embedding stage, and relative position encoding, because absolute encoding does not work well on longer sequences of input data. Along with these, the paper used various architectures like Linear RNNs, transformer, and Mamba for better performance.

![Performance comparison of Mamba and other language models on The Pile benchmark dataset](/blog/llms-always-hallucinate-paper-review/image_2.png)

Above figure demonstrates the performance comparison of Mamba and Other Language Models on The Pile Benchmark Dataset. Mamba exhibits comparable or slightly superior performance to other language models across various metrics on The Pile.

The paper also uses parameter efficient finetuning. Traditional fine-tuning updates all model parameters, which is costly for large models. Parameter-Efficient Fine-Tuning (`PEFT`) reduces computational expense by updating only a small number of parameters, helping models adapt to specific tasks. Adapters are a key `PEFT` method, introducing small trainable modules that are fine-tuned instead of the entire model. This approach preserves accuracy while lowering costs.

![Parameter-efficient fine-tuning with adapters — small trainable modules inserted into a frozen backbone](/blog/llms-always-hallucinate-paper-review/image_3.png)

After these, paper comes to the point that LLM hallucinations can occur even with the best training, fine-tuning, or the use of techniques like Retrieval-Augmented Generation (`RAG`) and it identifies four main types of hallucinations.

1. **Factual incorrectness:** occurs when LLMs provide incorrect information based on existing data, but without inventing new, non-existent details.
2. **Misinterpretation:** occurs when LLMs fail to correctly understand input or context, leading to inaccurate responses.
3. **Needle in a Haystack:** problem refers to the challenge LLMs face in retrieving specific, correct information from a vast corpus.
4. **Fabrications:** involve the creation of entirely false statements that have no basis in the model's training data.

Paper then uses techniques that attempt to improve, on every stage in the LLM, output generation process and hence mitigate hallucinations.

1. **Chain-of-Thought (CoT) prompting:** encourages LLMs to make the reasoning process explicit and potentially reduce logical inconsistencies and hallucinations. `CoT`, according to the paper, may explicit the process of reasoning, and help in reduction of logical errors and hallucination, it still does not eliminate hallucination entirely.

2. **Self-consistency:** is based on generating multiple reasonings using `CoT` and then select the most consistent one. The model is prompted with `CoT` and it generates multiple outputs for each reasoning, then it pairs each reasoning with its respective output and then selects the most consistent answer.

3. **Uncertainty quantification:** uncertainty quantification depends upon the model used. It basically helps us find out those instances where the model might be hallucinating.

    - **Softmax Neural Networks:** these are classifiers (algorithms) which categorize the data and predict the output based on softmax function. Softmax function calculates the probability distribution. Probability distribution tells the likelihood of different outputs. In this context, it tells how certain a model is in predicting an output and helps in uncertainty quantification.

    - **Bayesian Neural Network:** it treats weights as random variables that follow a probability distribution. Then a posterior probability distribution of weights is to be found given the input data. To quantify the uncertainty, the network uses a sample of weights drawn from posterior probability distribution to create multiple models. The final output is determined by averaging the prediction of these models and uncertainty can be quantified.

    - **Ensemble Neural Networks:** are just like Bayesian Neural Network, and the algorithm consists of a set of models as well. However, these models are independent of each other and make their predictions separately from other members of the set.

4. **Faithfulness:** it refers to the extent to which an explanation accurately reflects a model's reasoning process. It can be measured by Shapley Values. It measures how much each feature contributed to the model's predictions. Features which have high influence on output will have high Shapley value. This helps in understanding why the model reaches certain specific outputs.

The paper shows that no matter which one of the above techniques is applied, the fact remains the same — LLMs will hallucinate; hallucinations can never be fully eliminated. At every one of these stages, the LLMs are susceptible to hallucinations. Training can never be 100% complete; intent classification and information retrieval are undecidable; output generation is necessarily susceptible to hallucination; and post-generation fact checking can never be 100% accurate. Irrespective of how advanced our architectures or training datasets, or fact-checking techniques may be, hallucinations are still there.

![Hallucinations persist across every stage of the LLM pipeline](/blog/llms-always-hallucinate-paper-review/image_4.png)

Paper then explains some preliminaries it used to prove the claims that hallucination can never be fully eliminated. Firstly, Turing machines to address questions about determining mathematical proofs mechanically. Turing Machines can simulate any algorithm, making them useful for modelling LLMs. A Universal Turing Machine can simulate any other Turing Machine. Secondly, decidability, which means no matter how hard we try or how many resources we dedicate to the computer, a computer cannot solve all problems. The idealization of the Turing Machine helps in investigating the computations' limitations.

Now there are total of five claims, that we can never eliminate hallucination, the paper proves them using Turing machine algorithms.

1. **No training dataset can contain all true facts and no training data can ever be complete.** We can never give 100% a priori knowledge. The vastness and ever-changing nature of human knowledge ensures that our training data will always be, to some degree, incomplete or outdated.

    ![Gödel-like argument showing true facts always exist outside any finite training database](/blog/llms-always-hallucinate-paper-review/image_5.jpeg)

    With the help of an arbitrary LLM and a Gödel-like statement, it has been proved that there exist true facts beyond any finite training database. No matter how large our fact-checking dataset is, there will always be true statements that it does not contain. This inherent incompleteness contributes to the impossibility of eliminating all hallucinations by training the model on every possible fact.

2. **Even if the training data were complete, LLMs are unable to retrieve the correct information with 100% accuracy deterministically.** It means that if we assume that the training data set is 100% complete, correct, and contains all the facts about the world, and has up-to-date information, LLMs still cannot retrieve all the correct information with full accuracy. The paper calls it a "needle in a haystack" problem. Means, if an LLM is instructed to retrieve particular information (needle) from the complete training dataset (the haystack), LLM might possibly blur some information or data points and come up with the inaccurate information retrieval.

    > Note: please check the proof in the paper — it's easy to follow.

3. **An LLM will be unable to accurately classify intent with 100% probability.** The needle in a haystack problem leads to another problem: which is intent classification. We, humans, understand the intent and context and make correct and meaningful inference. But LLMs lack this ability and they are unable to reason. They are very vulnerable to ambiguities in user instructions as well as their knowledge system. Given the numerous possible interpretations of statements in natural language, there is always a non-zero probability that the model will retrieve the incorrect interpretation.

    > Note: please check the proof in the paper — it's easy to follow.

4. **No pre-training can deterministically and decidedly stop a language model from producing hallucinating statements.** It means we can improve the quality and quantity of training data, we can improve information retrieval, and if we improve intent classification but this leads to another problem. Language models don't know when to stop the generation process — their halting problem is undecidable. Therefore, LLMs are unable to know what exactly they will generate, as well as LLMs cannot check what they will generate before they actually generate. Then it means LLMs can hallucinate as well. Check proof in the paper.

5. **No amount of fact checking can completely eliminate every possible hallucination.** Even if the data is complete, retrieval system and intent classification are 100% accurate, as well as halting is decidable, the paper still claims that, if we verify each response of an LLM with a fact-checking database, we still cannot eliminate hallucination, no matter how big a fact-checking database is used. Proof is in the paper.

## Conclusion

This paper gave us very key information about LLMs. That if we apply different architectures, we apply different parameters and hyper-parameters finetuning techniques, we apply different mitigation strategies, as well as, in the ideal case, if we make training data 100% complete, 100% accurate `RAG`, and fact-checking mechanisms, LLMs will still generate incorrect and factless outputs. Which means that hallucination is the inherited problem, it lies in their fundamental mathematical structure, and we have to live with it.

## Personal Take

I suspect that one possible way that we can play with the mathematics now is to work closely on the attention mechanism — meaning rewind this concept, and start from the beginning. And we, at each stage, analyze and use different theorems, activation functions, and mathematically optimize these algorithms. Also, vector embeddings might need reconsideration and how can we mathematically optimize the accuracy of these positional encodings. These techniques might help in eliminating the hallucination.

[Link to the paper →](https://arxiv.org/abs/2409.05746)
