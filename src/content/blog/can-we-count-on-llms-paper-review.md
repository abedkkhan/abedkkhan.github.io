---
title: "What I learned from the Paper: Can We Count on LLMs? The Fixed-Effect Fallacy and Claims of GPT-4 Capabilities — A personal Take"
description: "A personal take on a paper that quantifies GPT-4's capabilities on trivial tasks, arguing we can't generalize LLM performance because minor prompt and input changes create variance beyond sampling noise."
pubDate: 2024-08-30
readTime: "9 min read"
tags: ["llms", "gpt-4", "paper-review", "evaluation"]
---

![Cover image — Can we count on LLMs? paper review](/blog/can-we-count-on-llms-paper-review/cover.webp)

This paper is under blind review, submitted to the journal "Transactions on Machine Learning Research." It aims to quantify the capabilities of LLMs, specifically `GPT-4`, on trivial tasks. The study investigates whether `GPT-4`'s responses are consistent across different scenarios or change when the input data or the prompt is slightly altered. Capabilities are examined on very simple tasks, such as counting numbers from a list, finding the mean, median, or maximum number from the list, or performing multiplication tasks, like multiplying numbers from the list that contain two digits (e.g., `45`) up to five-digit numbers.

To explore this, a list of input data is provided to `GPT-4` using different prompts. Conversely, the same prompt is used for different lists of input data to check the model's response. The goal is to determine whether `GPT-4` gives the correct response and if the responses change when the prompt or input data is altered. Several conditions per task have been examined (`500` per condition) to identify any statistically significant differences.

A chi-square statistical test is employed to compare the observed results with the expected results. This test determines whether the differences between the observed responses are due to chance or if there is a relationship between the responses provided by `GPT-4`. "By chance" means that different responses occur randomly, as in rolling a dice and getting a different number each time without any influencing factor. A relationship between the responses would indicate that differences are due to specific factors, such as changing the prompt, altering input parameters, or factors inherent in `GPT-4`'s architecture or training data.

Common and trivial tasks are used to assess the capabilities of `GPT-4` because, with more familiar tasks, there is a high probability that the model will give an accurate response, having been trained on tasks like solving chess puzzles, writing poetry, or doing some coding. However, counting the elements of a list is something the model predicts as a deterministic task with a clear and correct answer. These easy tasks are also simple to verify.

The author used the term "fixed effect fallacy." Fixed effects are those variables or effects whose impact is constant or fixed throughout the experiment. Based on fixed effects, we generalize the results of our experiment. But in the case of LLMs, we sometimes fall into the fixed effect trap and generalize the outcomes of language models. For example, if two researchers want to predict the impact of inflation on GDP, the two main variables are GDP and inflation. Apart from this, what the two researchers ate for breakfast, the color of their socks, etc., have no significant effect on their outcome; these factors have a fixed effect, and they can generalize their results. But in the case of language models, tiny changes in the prompt, tiny changes in the input data, and very minor changes can impact the performance of LLMs. So, we cannot generalize the performance of LLMs because they are very complex in structure, and these minor changes and variations in input play a significant role.

To compute the margin of error, the author determines the degree to which the actual response of `GPT-4` differs from the predicted response. Initially, the assumption is made that the only cause of randomness is the variation in the sample size, using a `95%` confidence interval with a z-score of `1.96`. The margin of error is calculated as `1.96 √(p(1 − p)/N)`, where `p` is the success rate, and `q` is the failure rate. The success rate minus the failure rate essentially gives the sample proportion, which is the percentage of the sample that exhibits the characteristics the author is interested in, such as the percentage of correct responses, denoted by `p`. Therefore, `1-p` represents the percentage of incorrect responses (failure rate). In one specific trial, the author calculates the margin of error as `2.27` with a success rate of `89%`. This means that, out of `100` responses, `95` times the result will fall within the interval of `91.7%` (upper bound: `89 + 2.7`) and `86.3%` (lower bound: `89 – 2.7`), which constitutes the confidence interval. However, the author notes that in this calculation, the only cause of randomness considered is the variance in the sample, while other, more significant causes of randomness are not accounted for. Due to this, the author opts to keep only the lower bound on the margin of error, acknowledging that the actual upper bound could be much higher than the one calculated based on sample variance. This approach is also considered safer to avoid giving a false sense of precision.

![Trial 1 — counting task results under varying prompt wording and list lengths](/blog/can-we-count-on-llms-paper-review/image_1.png)

![Trial 1 — chi-square analysis of GPT-4 counting accuracy across conditions](/blog/can-we-count-on-llms-paper-review/image_2.png)

In this specific trial, the author uses four different conditions: changing the wording of the prompt, using different items in the list, and altering the length of the list. Results show that increasing the length of the list decreases the accuracy of `GPT-4`, changing the wording of the prompt while keeping everything else the same generates inaccurate and different responses, and different items in the list lead to different performance levels even though the wording of the prompt and the length of the list remain the same. Since the author uses the chi-square test to statistically verify the results, the null hypothesis is robustly rejected (`p-value < 0.05`), meaning that results from different conditions (e.g., changing the wording of the prompt) should be the same. Similarly, the null hypothesis is also rejected when changing the list items while keeping the prompt the same. Across `500` trials, the mean `GPT-4` responses are always lower than the correct answers. This indicates that minor and simple modifications in tasks, which might easily be assumed to make no difference, are actually sources of variance beyond what can be explained by sampling effects.

![Trial 2 — GPT-4 accuracy on max, median, and sorting tasks](/blog/can-we-count-on-llms-paper-review/image_3.png)

![Trial 2 — performance drop when list format changes from numeric to name-value pairs](/blog/can-we-count-on-llms-paper-review/image_4.png)

In the second trial, the author performs three basic operations on the list: finding the maximum, finding the median, and sorting the list. `GPT-4` performs well on the maximum task with `90%` accuracy, indicating that it was relatively easy for the model to perform this task. In contrast, `GPT-4` performs poorly in finding the median, especially when the list includes decimal numbers. This may occur when the list length is even and requires averaging the two middle numbers. Moreover, `GPT-4` performs well when sorting a list of simple numbers. However, when the author changes the list to a name-value pair, the performance drops drastically to `55%`. This suggests that changing the list type (e.g., from decimals to integers or to a name-value pair) introduces variations that impact performance beyond what random sampling would explain.

![Trial 3 — GPT-4 multiplication accuracy across digit lengths](/blog/can-we-count-on-llms-paper-review/image_5.png)

In the third trial, `GPT-4`'s ability to perform long multiplications has been evaluated. The task asks `GPT-4` to multiply two numbers, with the number of digits varying in size:

- **2 × 2 multiplication:** Both numbers have 2 digits.
- **3 × 3 multiplication:** Both numbers have 3 digits.
- **4 × 4 multiplication:** Both numbers have 4 digits.
- **5 × 5 multiplication:** Both numbers have 5 digits.
- **Mixed-length multiplication:** For example, multiplying a 4-digit number by a 2-digit number.

The author finds that `GPT-4` performs well on smaller multiplications, such as `2 × 2` (with `100%` accuracy) or `3 × 2` (`91.6%` accuracy), but its performance degrades significantly for larger multiplications like `4 × 4` (`3.2%` accuracy) or `5 × 5` (`0%` accuracy). Changing the length of the numbers and the order of multiplication also impacts `GPT-4`'s performance, beyond what can be explained by the variation in sample size, which is the assumption in calculating the margin of error.

## Conclusion

One of the main outcomes of this paper is that we cannot generalize the performance of LLMs. There are no fixed effects that can be ignored or deemed insignificant. The author demonstrates this through several different conditions and sufficient statistical power to rule out sampling noise as the sole source of variation. This allows the author to confidently state that minor modifications can have potentially enormous effects on measured capabilities. This problem is entirely orthogonal to the frequently mentioned issue of hallucinations.

## My Personal Opinion

This paper provides a good foundation for further research on LLM capabilities. However, we cannot generalize the findings to all LLMs, as this paper only examines `GPT-4`. We have a variety of LLMs available, and a comparative analysis of these models could provide more comprehensive insights. It is important to investigate whether the issues observed are due to the architecture, training data, or predictive power of the models. Additionally, we should confirm whether these issues persist across different versions of GPT, such as `GPT-2`, `GPT-3`, `GPT-4`, and `GPT-4o`, or if they are specific to `GPT-4`. We can also verify the findings with different LLM architectures, like `Mamba-2` and `Jamba`, to determine if architectural innovations can help mitigate the generalization error of LLMs. Finally, we need to understand the major sources of performance in LLMs, such as data scale and model size, and explore whether scaling helps address these issues.

[Link to the paper →](https://openreview.net/forum?id=qt4d0EGZsK)
