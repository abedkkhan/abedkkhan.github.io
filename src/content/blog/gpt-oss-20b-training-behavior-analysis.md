---
title: "What GPT-OSS20b Outputs Tell Us About Its Training and Behavior"
description: "Probing GPT-OSS-20B without the Harmony format — 1M inferences, cosine-distance drift analysis, and what the model's collapse into code/math tokens reveals about its training data."
pubDate: 2025-08-21
readTime: "14 min read"
tags: ["gpt-oss", "llms", "training-data", "embeddings", "mechanistic-interpretability"]
---

![Cover image — GPT-OSS-20B training and behavior analysis](/blog/gpt-oss-20b-training-behavior-analysis/cover.webp)

OpenAI stated in the [GPT-OSS documentation](https://github.com/openai/gpt-oss?utm_source=chatgpt.com) that these models should only be used with the Harmony response format; otherwise, they won't work correctly. They're trained on this specific format, only understand this, and only respond properly when used with the Harmony format. This raises a couple of questions:

- What exactly does "won't work properly" mean?
- What is it about the Harmony response format that makes the model behave correctly?
- How heavily are these models trained on the Harmony format, such that they shift from being raw next-token predictors to behaving like structured conversational models?
- And if we skip Harmony, does that basically reduce the model to raw completion mode? If so, how well can it still predict the next token?

Jack Morris, in [one of his tweets](https://x.com/jxmnop/status/1953899426075816164), generated about 10M responses from GPT-OSS-20B, most likely without using the Harmony protocol. He ran some analysis and claimed the model shows a strong bias towards math and code. In fact, for many general prompts, it tends to drift back into math or programming responses. This could suggest that the model (and maybe even GPT-OSS-120B) was trained heavily on math/code domains or benchmarks. He also argued that by generating this kind of large-scale responses, we can get a rough glimpse into the underlying training data of the model.

In this blog, we'll try to replicate and verify Morris's findings, and, in the process, maybe we'll get closer to answering the questions raised above.

For this experiment, I'm using Ollama to run GPT-OSS-20B locally. We'll be using the model without the Harmony format, just to see how it behaves in plain mode. The assumption here is simple: if we don't use Harmony, the model falls back into being a typical completion model, just predicting the next token without any structured conversation format.

The hypothesis is: if we give it a single word, or even an empty prompt, in what "direction" does it start predicting the next token?

- **Null hypothesis (H₀):** when given a general one-word prompt, the model does not consistently continue in the same direction.
- **Alternative hypothesis (H₁):** when given a general one-word prompt, the model does predict the next token in the same direction.

By "same direction" we mean, if we prompt with something like "quantum mechanics," the model's next predicted tokens stay in that domain — e.g., quantum mechanics, physics, scientific concepts.

If the null holds true, then it strengthens Jack Morris's claim: that these models' training data is biased, with GPT-OSS being heavily trained on math and code, to the point that even general prompts drift back to those domains. Also, it may not have real-world general understanding that much.

## Inferences

With that setup, after getting the model running via Ollama locally, we define our prompts as follows:

```python
input_prompts = [
    "Smile", "Joy", "Sadness", "Anger", "Fear", "Love", "Hope", "Peace", "Calm", "Rage",
    "Happy", "Excited", "Nervous", "Confident", "Worried", "Grateful", "Lonely", "Pride",
    "Apple", "Car", "House", "Tree", "Ocean", "Mountain", "Book", "Phone", "Computer",
    "Chair", "Table", "Window", "Door", "Key", "Lamp", "Mirror", "Painting", "Clock",
    "Flower", "Stone",
    "Run", "Jump", "Dance", "Sleep", "Think", "Write", "Read", "Listen", "Watch", "Create",
    "Build", "Destroy", "Help", "Learn", "Teach", "Play", "Work", "Rest", "Travel", "Explore",
    "Freedom", "Justice", "Truth", "Beauty", "Wisdom", "Courage", "Honor", "Faith", "Trust",
    "Mystery", "Future", "Past", "Present", "Infinity", "Nothing", "Everything", "Reality",
    "Dream", "Memory", "Imagination",
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Black", "White", "Pink", "Brown",
    "Cat", "Dog", "Bird", "Fish", "Lion", "Tiger", "Elephant", "Horse", "Rabbit", "Snake",
    "Pizza", "Cake", "Bread", "Milk", "Coffee", "Tea", "Rice", "Pasta", "Soup", "Salad",
    "A", "B", "C", "X", "Y", "Z",
    "One", "Two", "Five", "Ten", "Hundred", "Thousand", "Million", "Zero",
    "", " ", ".", "?", "!", "...", "???",
    "The", "And", "Or", "But", "If", "When", "Why", "How", "What",
    "Quantum", "Gravity", "Energy", "Matter", "Space", "Time", "Evolution", "DNA", "Atom",
    "Universe", "Existence", "Consciousness", "Soul", "Mind", "Body", "Spirit", "Ethics",
    "Morality", "Purpose", "Meaning",
]
```

After defining prompts, we define the inference function, and we also define how many inferences we want the model to make; in this case, I went for 1 million, 1k per file. Also, the [`raw` flag](https://github.com/ollama/ollama/blob/main/docs/api.md?utm_source=chatgpt.com) is `True`, which means no Harmony formatting will be applied to the prompt.

```python
def make_inference(prompt):
    """Send one prompt to the model and return text or error string."""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gpt-oss:20b",
        "prompt": selected_prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 4000,
            "num_ctx": 2100,
            "top_p": 0.9,
        },
    }


total_inferences = 1_000_000
inferences_per_file = 1000
```

Now, we run the loop to randomly select a prompt from `input_prompts` and make 1M inferences, saving both the input query and output response in a clean way for further analysis.

```python
for i in range(total_inferences):
    selected_prompt = random.choice(input_prompts)
    file_number = (i // inferences_per_file) + 1
    output = make_inference(selected_prompt)
    save_inference(selected_prompt, output, file_number)
    if (i + 1) % 100 == 0:
        print(f"Completed {i+1:,} inferences ({((i+1)/total_inferences)*100:.2f}%)")
    time.sleep(5.0)
```

We have created 1M inferences. A screenshot of 1 in a million is following — where GPT-OSS-20B, instead of answering about physics or Newton's laws, coded an entire JavaScript game character with jump mechanics and all, lol.

![One-in-a-million sample — GPT-OSS-20B answering a physics prompt with a full JavaScript game character](/blog/gpt-oss-20b-training-behavior-analysis/image_1.png)

## Embeddings

After generating the inferences, we take the input query and its corresponding output and create embeddings for both. By computing the cosine distance between them, we can measure how semantically close or far the output is from the original query.

To dig deeper, we split the outputs into chunks of 400 words with a 100-word overlap. Then, for each sentence in the output, we create embeddings and calculate its distance from the input query. This way, we can track not just the overall closeness but also how the output drifts in meaning as the prediction goes on.

For example, with a query like "smile," the model might start with something loosely related — say, about time or emotions — and then gradually deviate into other domains like math or code. By analyzing sentence-level distances, we can see exactly when the output stays aligned with the query and when it strays off-topic.

For embeddings, we use OpenAI `text-embedding-3-small`.

```python
def get_openai_embedding(text, model="text-embedding-3-small"):
    resp = openai.embeddings.create(input=text, model=model)
    return np.array(resp.data[0].embedding)


def chunk_text(long_sentence, chunk_size=400, overlap=100):
    """Split by words into overlapping chunks (same pattern you used)."""
    words = long_sentence.split()
    chunks, start = [], 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        if end >= len(words):
            break
        start += step
    return chunks
```

After setting up the embedding and chunking functions, we also define some simple parsing and data-loading utilities. These are included in the Jupyter notebook linked in the GitHub repo attached to this blog.

Once the parsing is done, as shown in the code below, we generate embeddings for the input query as well as for each sentence in the model's output in sequence (Sentence 1, Sentence 2, and so on). Alongside the embeddings, we also store the raw sentences themselves. This is important, because later we can attach labels or domain/topic names to each sentence.

```python
for i, rec in enumerate(records, 1):
    query = rec["input"]
    output_joined = " ".join(rec["output"].split())

    # 1) input embedding
    q_emb = get_openai_embedding(query)
    with open(input_emb_file, "a", encoding="utf-8") as f:
        f.write(f"Embedding: {q_emb.tolist()}\n")

    # 2) output → chunks (sentences)
    chunks = chunk_text(output_joined, chunk_size=400, overlap=100)

    # save the sentences
    with open(chunks_text_file, "a", encoding="utf-8") as f:
        for j, ch in enumerate(chunks, 1):
            f.write(f"Sentence {j}: {ch}\n")

    # 3) embeddings for each sentence
    with open(output_emb_file, "a", encoding="utf-8") as f:
        for j, ch in enumerate(chunks, 1):
            emb = get_openai_embedding(ch)
            f.write(f"Inference {i} | Sentence {j}: {emb.tolist()}\n")
```

For this case, I used only 10 inferences to create embeddings, though the approach can be extended further.

## Cosine Distance

After storing the embeddings, we calculate the cosine distance between the query and each sentence in its output. This distance gives us a way to quantify the semantic gap between prompt and response.

The setup is simple: basic imports and input parsing utilities are already defined in the Jupyter notebook (linked in the repo). After running those, the main function is shown below.

```python
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(a @ b / denom) if denom else 1.0


for inf in range(1, 11):
    q = extract_query_embedding(inf)
    sents = extract_sentence_embeddings(inf)

    out_path = os.path.join(output_dir, f"inference{inf}_cosine_distances.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("sentence_index,cosine_distance\n")

    print("Saved:", out_path)
```

Hence, we compute the cosine distances and save them.

## Classification

Cosine distance alone may not tell a full story. So, the next step is to classify each output sentence and assign it a label or topic, showing what the sentence is about and which domain it belongs to. For this classification and labeling, we use GPT-OSS-20B with the Harmony format (this time with the `raw` flag set to `False`).

Since we already stored the input queries and output sentences, the parsing, data-loading, and saving utilities (defined in the Jupyter notebook) handle the setup in the right order. Below is the exact prompt we use for labeling and classification.

```python
async def generate_simple_label(sentence_text):
    try:
        prompt = f"""You are an expert text analyzer. Your task is to read the following text and create a precise, descriptive label that captures its essence.

TEXT: "{sentence_text[:300]}"

HOW TO ANALYZE:
1. First, identify the MAIN SUBJECT or topic being discussed
2. Second, determine the FIELD or DOMAIN this belongs to
3. Third, consider the SPECIFIC CONTEXT or approach being used
4. Finally, create a label that combines the most important aspects

THINKING PROCESS:
- What is this text primarily about?
- What field of knowledge does this belong to?
- What specific aspect or angle is being discussed?
- How would an expert in this field categorize this?

LABEL REQUIREMENTS:
- Use EXACTLY 2-3 words
- Be specific and descriptive
- Use clear, professional terminology
- Capture the most important essence of the text

EXAMPLES OF GOOD LABELS:
- "quantum mechanics" (not just "physics")
- "data structures" (not just "programming")
- "financial modeling" (not just "business")
- "cognitive psychology" (not just "psychology")
- "organic synthesis" (not just "chemistry")
- "machine learning" (not just "technology")

CREATE YOUR LABEL (2-3 words): """
```

After labeling each sentence, we move on to visualizing the results. The visualization code is included in the Jupyter notebook.

## Visualization

![Figure A — cosine distance between each query and the sentences in its output, across 10 prompts](/blog/gpt-oss-20b-training-behavior-analysis/image_2.png)

Figure A shows the set of queries we used in this experiment. The x-axis represents the cosine distance. We deliberately chose queries that are very general and tied to everyday language, with nothing from programming, math, or logic domains. Each dot represents one sentence (or chunk) of the model's output, plotted in order from left to right.

Take the query "Spirit" for example, on the bottom: the output has two sentences, one landing just above 0.80 cosine distance and the second around 0.85. Looking across all queries, the first big observation is that none of the output sentences are semantically very close to their input query. The minimum distance we see starts above 0.60. This suggests that when GPT-OSS-20B is given a one-word, general-life prompt, the next tokens it predicts already sit at least 0.60 distance away from the query.

The second insight is about drift. As generation progresses, cosine distance keeps increasing. In other words, the further the model goes, the further away it gets from the meaning of the original query. It's still predicting tokens with high probability, but those tokens may not necessarily stay related to the input.

So across these 10 examples, the model consistently moves semantically away from the prompt instead of staying attached. That means if we give GPT-OSS-20B a general life query, its responses are never truly close to the query — at best loosely related.

This naturally raises the next question: if the outputs keep drifting away, what exactly is the model predicting instead? We dig into that in the next section (see the table below).

![Table of labels — each output sentence classified by domain, showing drift into code/math](/blog/gpt-oss-20b-training-behavior-analysis/image_3.png)

The table above labels each output sentence in the exact left-to-right order shown in Figure A. This lets us see what the model is drifting into as the generation progresses.

What stands out immediately is that the outputs are not grounded in the input query at all. For example:

- The query "Spirit" quickly turns into "Java game objects," "Thread-safe inventory," and "Java concurrency."
- "Smile" ends up with "prefix lookup" and "database query performance."
- "Anger" shifts into "synthetic emotion embeddings" and "logic."
- Even "Pasta" gives back "PyQt GUI."
- "Excited" jumps straight into "quantum state estimation" and "statistical modeling."

Across all 10 cases, the same pattern holds: general, real-life queries are hijacked by technical or programming-related completions. The drift we saw in cosine distance (Figure A) is now explained: the model is not staying close to the original concept but rather collapsing into code, software, or math tokens that it assigns high probability.

So the combined picture is:

- Figure A shows semantic drift (increasing distance from the query).
- This table shows the direction of that drift (logic, programming, technical jargon).

Now, the above results circle back to the questions we raised in the beginning. First of all, we did not use GPT-OSS-20B with the Harmony format, and it did not work correctly. It wasn't just that the answers were wrong — the model went in the exact opposite direction of the query. That alone suggests something deeper: this model seems to be heavily trained on, or even locked into, the Harmony format. If you don't follow that protocol, the model basically breaks. It doesn't know what to do, and in that confusion it starts spitting out random completions, often technical and often completely unrelated to the input. That dependency is not just an artifact; it raises a vulnerability. If a model is 100% relying on a strict prompting protocol, then its safety guardrails might also be fragile. Maybe it can be tricked, bypassed, or exploited if someone knows how to deliberately break that Harmony flow. The fact that the exact same model, when used with Harmony in labeling, suddenly performs so well only strengthens that suspicion.

Another thing: we assumed that using the model without Harmony might correspond to some kind of "completion mode," where the model just predicts the next tokens based on the input. But in our case, although it was predicting next tokens, those predictions were not semantically close to the query at all. That raises another question: is this actually the base/completion version of the model, or is GPT-OSS a highly instruction-tuned model, and what we saw was just a mismatch between what we expected and what the model is designed for?

Lastly, our analysis does give some weight to Jack's claims — that these models collapse into domains of code and math. When we gave it simple, general queries, the responses drifted back to coding and technical jargon. Does this mean the model has been heavily trained, maybe even over-trained, on math, programming, and benchmarks, to the point where it doesn't have much understanding of the general world? And if this is the case, then when we run large-scale probing experiments — generating millions of responses, analyzing deviations, and tracking what direction outputs take — maybe we can use this drift as a way to uncover the true nature of a model's training data.

This blog leaves us with even more questions than we started with. I plan to extend these experiments to other models and see what kind of stories they tell. I'd love for the above results to be replicated, and I'm looking forward to hearing insights from others. Let's see if we can start answering some of these questions and understand these models better.

[GitHub Repo](https://github.com/abedkkhan/gpt_oss_testing)
