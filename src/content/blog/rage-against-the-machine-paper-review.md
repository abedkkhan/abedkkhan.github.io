---
title: "Analysis of the Paper: RAGE Against the Machine — Retrieval-Augmented LLM Explanations. A personal Take"
description: "A personal take on the RAGE paper: how combinations and permutations of retrieved sources change LLM answers, and how RAGE exposes the 'lost in the middle' problem."
pubDate: 2024-09-14
readTime: "10 min read"
tags: ["llms", "rag", "paper-review", "retrieval-augmented-generation"]
---

In recent years, we've witnessed remarkable advancements, particularly with large language models like `GPT`, `Gemini`, `Claude`, and others. Research scientists are now training increasingly complex models, extending their context lengths, and equipping them with billions or even trillions of parameters and boosting their computational capabilities. However, as these models improve rapidly, issues arise regarding their outputs. Researchers are now focusing on quantifying the accuracy, timeliness, and reasoning behind these outputs, as well as troubleshooting their limitations.

In [my last blog post](/blog/can-we-count-on-llms-paper-review/), I discussed a paper exploring whether we can generalize LLMs, rely on their capabilities, and avoid falling into the trap of fixed effects.

One of the key advancements that sets modern Large Language Models (LLMs) apart from previous iterations is the introduction of Retrieval-Augmented Generation (`RAG`). `RAG` is a powerful prompt engineering technique that enables LLMs to access and incorporate external sources of information, which are directly supplied via the prompt. While LLMs come with a vast amount of pre-trained knowledge, this knowledge may become outdated or irrelevant for certain user queries. `RAG` addresses this limitation by allowing LLMs to retrieve up-to-date and contextually relevant information from the internet or other external databases. The model then combines this real-time data with its pre-trained knowledge, generating a more accurate and informed response tailored to the user's query. `RAG` has been very useful for LLMs to reduce their hallucination problem to generate plausible yet incorrect outputs. But there are some limitation of this powerful advancement as well.

The paper *RAGE Against the Machine: Retrieval-Augmented LLM Explanations* explores the limitations of Retrieval-Augmented Generation (`RAG`) models, addressing them both mathematically and through a graphical demonstration tool. The authors highlight several concerns with `RAG` models, including:

1. **Lack of transparency in the origin of responses:** It's often unclear where the LLM has retrieved its resources from, raising questions about how updated and relevant these sources are to the user's query. This lack of traceability can make it difficult to assess the accuracy and reliability of the response.

2. **"Lost in the middle" context issue:** When responding to a user query, `RAG`-based LLMs sometimes prioritize documents listed at the top or bottom of the retrieval list, ignoring sources that appear in the middle. This can lead to incomplete or biased responses if key information is missed.

To address these concerns, the paper introduces the `RAGE` tool, which helps analyze the influence of minimal changes in the combination and permutation of retrieved sources on the LLM's response. By experimenting with different source configurations, the tool aims to understand how these variations impact the quality of the LLM's output.

The paper further explains that the `RAGE` tool is fully compatible with any transformer-based LLM. It retrieves all knowledge sources (documents) from locally configured document indexes using the `BM25` retrieval model from the `Pyserini` retrieval toolkit, allowing a controlled and consistent retrieval process for analyzing `RAG` performance.

![RAGE tool overview — interface for analyzing retrieval-augmented LLM responses](/blog/rage-against-the-machine-paper-review/image_1.png)

In the `RAGE` users can generate explanations of LLM responses through two methods: source combinations and source permutations.

1. **Source combinations:** This approach explains how the presence or absence of certain resources affects the LLM's response. By adding or removing specific sources, users can observe how these changes impact the LLM's output. This helps in understanding which resources are most influential in shaping the response.

2. **Source permutations:** This method explores how changing the order of the retrieved resources alters the LLM's response, even when the prompt and context remain the same. By simply rearranging the order of the sources (e.g., moving a resource from top to bottom), users can observe how the model produces a different response despite no changes in the content of the sources themselves.

Additionally, `RAGE` allows users to visualize the distribution of answers based on various combinations and permutations, along with a detailed list of all the variations. The graphical representation helps users better understand how sensitive LLMs are to the selection and ordering of external sources during retrieval-augmented generation.

According to the paper, `RAGE` performs two key tasks, referred to as Top-Down changes and Bottom-Up changes. These methods aim to analyze how the presence or absence of external sources influences the LLM's response to a query.

1. **Top-Down changes:** This method starts with a complete set of retrieved sources denoted as `Dq` and systematically removes or adds certain sources to determine if the LLM's response changes. The purpose of this approach is to evaluate which sources are crucial or irrelevant for generating the response. By removing different sources one by one, it identifies the minimal subset of sources that, when excluded, leads to a different target response. For example, if the LLM gives an answer based on five sources, the top-down method would progressively remove some of these sources to see how the LLM's answer shifts. This helps in understanding which sources play a pivotal role in shaping the response.

2. **Bottom-Up changes:** This method begins with an empty set of external sources and incrementally adds sources to observe how the LLM's response evolves. Initially, the LLM might provide a generic or default answer in the absence of external information. The bottom-up approach then adds sources step by step, identifying the minimal combination of sources that lead to a change in the response, eventually reaching the desired target answer. This method highlights which sources are needed to influence the LLM's final response.

Since there are multiple combinations of equal-sized sources, their relevance scores have been calculated and are expressed as `∑ = S(q, d, Dq)`, where `q` is the query, `Dq` is the set of all retrieved resources, and `d` is a subset of `Dq`. The relevance score is important for determining which specific sources contributed to the LLM's response. According to the paper, users can choose between relevance scores calculated by the LLM or those calculated by `RAGE`.

In simpler terms, the model keeps track of how much attention it gave to each source during its internal processing. The aggregation process involves summing the attention values over all layers and attention heads in the model, which means the attention values are accumulated across every layer and attention head to provide a comprehensive view of the model's focus. Additionally, the attention is summed over the tokens corresponding to each source, meaning the model calculates the total attention given to the specific tokens (or words) that belong to each source in the combination. While, the retrieval model picks which external knowledge sources to include in the LLM's input context by giving each source a relevance score based on how well it matches the query. This score tells how close the source is to the user's question.

Now, there is one simple algorithm used on permutation and combination to find when the response of LLM changes. Algorithm generates all length-k permutations for the `k` sources, then computes Kendall's Tau rank correlation coefficient for each permutation (with respect to their given order in `Dq`). Once generated and measured, the permutations are subsequently sorted and evaluated in decreasing order of similarity, based on decreasing Kendall's Tau. In the same way it works for combination. To supplement this counterfactual analysis, answers are analyzed over a selected set of perturbed sources. To obtain a set of combinations, `RAGE` considers all combinations of the retrieved sources `Dq`, or draws a fixed-size random sample of `s` combinations. Based on the user's original question, a prompt is created for each selected combination, which is then used to retrieve corresponding answers from the LLM. After analyzing the answers, `RAGE` renders a table that groups combinations by answer, along with a pie chart illustrating the proportion of each answer across all combinations. A rule is determined for each answer, when applicable, identifying sources that appeared in all combinations leading to this answer.

In `RAGE`, users can analyze how different permutations of sources affect LLM answers. Users can look at all possible permutations or a fixed-size random sample. Instead of generating all possible permutations, which is time-consuming, `RAGE` uses a more efficient method with the Fisher-Yates shuffle algorithm to quickly get random permutations. `RAGE` also helps find the best permutations by placing important sources in positions where the LLM pays the most attention. Users can choose to use either the LLM's attention scores or the retrieval model's relevance scores for this analysis. `RAGE` uses a smart algorithm to find the top permutations efficiently, focusing on both the relevance and the attention of the sources.

Above, is the basic working of `RAGE`, which tells us how LLMs being so powerful, generate outputs on very trivial tasks, just changing the combinations or permutation. Below are two of the examples explained in the paper, about how changing permutation and combination of resources alters the response. For checking all the examples and to try this tool, please refer to the paper given below.

## Example 1

A user asks an LLM to identify the best tennis player among Novak Djokovic, Roger Federer, and Rafael Nadal, using documents that rank them based on different metrics. Despite Djokovic having more Grand Slam wins, the LLM selects Federer. The user then uses `RAGE` to analyze which documents influenced this choice. `RAGE` finds that a document ranking Federer first was crucial in the LLM's decision. The user notices this document was prominently placed at the start. To see how position affects the result, they use `RAGE` to test different placements and find that moving the document changes the answer to Djokovic. This analysis helps the user understand which document influenced the LLM's response and how its position affected the outcome.

![Example 1 — initial document ordering leads LLM to pick Federer as the best player](/blog/rage-against-the-machine-paper-review/image_2.png)

![Example 1 — reordering the same documents changes the LLM's answer to Djokovic](/blog/rage-against-the-machine-paper-review/image_3.png)

## Example 2

A user checks the latest US Open women's tennis winner using an LLM and gets "Coco Gauff" as the answer. To verify, they look for the source document behind this result and find Gauff listed as the 2023 champion in the last document. Wondering if outdated documents might lead to errors, they use `RAGE` to reorder the context documents. When the last document is moved, the LLM wrongly picks 2022 champion Iga Swiatek. This helps the user see that the final document was the only up-to-date one, while older ones caused confusion.

## Personal Take

LLMs rely heavily on their self-attention mechanism, which allows them to assign different weights to various parts of the input based on relevance. However, this can lead to biases where the model prioritizes sources at the beginning or end of the context. To improve this, we can fine-tune the self-attention mechanism by introducing regularization that ensures a more balanced distribution of attention across all sources, preventing important middle-positioned data from being overlooked.

In retrieval-augmented models like `RAG`, the retrieval mechanism selects external sources based on relevance, but these selections may not always align with the LLM's decision-making process. A solution would be to jointly train both the retrieval mechanism and the LLM so that the model learns to make better, more consistent decisions based on the selected sources, reducing the sensitivity to small changes in input.

Finally, LLMs struggle with maintaining context over long inputs, often forgetting or overlooking important information. Implementing long-term memory mechanisms or hierarchical attention could allow the model to retain crucial details over extended contexts, ensuring better understanding and processing of long-form inputs without losing important information.

[Paper link →](https://arxiv.org/pdf/2405.13000)
