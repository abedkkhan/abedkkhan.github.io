---
title: "Recursive Language Models: When Context Windows Aren't Enough"
description: "A personal experiment trying to replicate Recursive Language Models — treating prompts as external data the LLM explores through code and recursive sub-calls — and what it taught me about context rot."
pubDate: 2026-01-08
readTime: "11 min read"
tags: ["llms", "rlm", "context-window", "gpt-4o", "experiments"]
---

![Cover image — Recursive Language Models experiment](/blog/recursive-language-models-experiment/cover.webp)

I recently came across a paper called *"Recursive Language Models"* that made a claim I couldn't ignore — we can make LLM handle inputs longer than its context window, and not just handle them, but actually perform better on complex tasks.

We don't need to train a new model. We don't need special hardware. We just need to change how we use the model we already have.

Here's the core idea: instead of stuffing a massive prompt directly into the LLM, we treat the prompt as external data, like a variable in a Python environment. The LLM can then write code to look at it, filter it, search through it, and recursively call smaller versions of itself to process chunks. It's like giving the model a desk, a computer, and a team of assistants, instead of asking it to memorise an entire library.

When I read this, I thought: "This is smart. The concept is simple enough that I could probably replicate it in an afternoon."

So I tried.

Spoiler: I didn't fully succeed. Building a working Recursive Language Model turned out to be a bit trickier than the paper made it look. But the experiment taught me something valuable about both the power and the practical challenges of this approach, and confirmed that the problem it's trying to solve is very real.

## The Problem — Context Rot is Real

Before diving into my experiment, let me explain the problem that Recursive Language Models are trying to solve.

We know that modern LLMs have massive context windows — `GPT-4` can handle around `128K` tokens, `Claude` can around `200K`, and some models claim even more. So long context problems should be solved, right?

Not quite.

The paper introduces a concept called "context rot" — even when a document (input or prompt) technically fits within an LLM context window, the model performance degrades as the input gets longer. It's not just about whether the tokens fit, it's about whether the model can actually use them effectively.

Think of it like this: you can technically read a `300`-page book in one sitting, but by page `250`, you're probably forgetting details from page `50`. LLMs have somehow the same problem.

Here's where it gets interesting — the paper argues that context rot isn't just about input length, it's also about task complexity.

Three types of tasks:

1. **Needle-in-a-Haystack (Constant Complexity)**
    - **Task:** Find one specific fact hidden in a huge document
    - **Complexity:** Doesn't matter how long the document is, you're still looking for one thing
    - **Result:** LLM handles this fine, even at `1M+` tokens

2. **OOLONG (Linear Complexity)**
    - **Task:** Aggregate information from almost every line of input
    - **Example:** "Count how many questions in this dataset are about locations vs. people"
    - **Complexity:** If input doubles, work doubles
    - **Result:** Performance starts degrading at much shorter lengths

3. **OOLONG-Pairs (Quadratic Complexity)**
    - **Task:** Reason about relationships between pairs of items
    - **Example:** "Find all pairs of users where both have X and Y properties"
    - **Complexity:** If you have `N` items, you have `N²` possible pairs to check
    - **Result:** LLM (`GPT-5`) performance collapsed

This is the problem RLMs are designed to solve — not just making inputs longer, but handling tasks where the information density is high, where we can't skip anything, and where relationships between pieces matter.

## My Experiment

I decided to create a test that would push a model into that "quadratic complexity" zone where the paper showed catastrophic failure.

I generated a synthetic dataset:

```python
from collections import defaultdict

categories = ['location', 'person', 'number', 'description', 'entity', 'abbreviation']

templates = {
    'location': ["Where is {}?", "How do I get to {}?", "What's the weather in {}?"],
    'person': ["Who is {}?", "What did {} accomplish?", "When was {} born?"],
    'number': ["How many {}?", "What's the count of {}?", "How much does {} cost?"],
    'description': ["What is {}?", "Explain {}", "Define {}"],
    'entity': ["What company makes {}?", "Who owns {}?", "Which brand sells {}?"],
    'abbreviation': ["What does {} stand for?", "What is {} short for?", "Expand {}"]
}

places = ['Paris', 'Tokyo', 'Berlin', 'Cairo', 'Sydney', 'Mumbai', 'Toronto', 'Rio']
people = ['Einstein', 'Tesla', 'Curie', 'Darwin', 'Newton', 'Galileo', 'Hawking']
things = ['apples', 'cars', 'books', 'stars', 'trees', 'phones', 'computers']
concepts = ['democracy', 'gravity', 'evolution', 'energy', 'freedom', 'justice']
products = ['iPhone', 'Windows', 'PlayStation', 'Kindle', 'Android', 'Chrome']
abbrevs = ['NASA', 'FBI', 'CIA', 'WHO', 'UN', 'EU', 'NATO', 'ASAP']

fillers = {'location': places, 'person': people, 'number': things, 'description': concepts, 'entity': products, 'abbreviation': abbrevs}

def gen_question(cat): return random.choice(templates[cat]).format(random.choice(fillers[cat]))

users = []
for uid in range(1000):
    n_questions = random.randint(3, 5)
    user_cats = [random.choice(categories) for _ in range(n_questions)]
    questions = [gen_question(cat) for cat in user_cats]
    users.append({'uid': uid, 'questions': questions, 'categories': user_cats})

context = "\n".join([f"User {u['uid']}: {', '.join(u['questions'])}" for u in users])
```

The setup:

- `1000` users
- Each user has `3-5` questions
- Questions belong to semantic categories: location, person, number, description, entity, abbreviation
- Categories aren't labeled — the model has to figure them out.

```
User 0: What's the weather in Rio?, Which brand sells PlayStation?, Where is Cairo?
User 1: How do I get to Tokyo?, Who owns Chrome?, What did Tesla accomplish?
User 2: Explain justice, What's the count of stars?, What does FBI stand for?
```

Total size: `~23,000` tokens. Easily fits in `GPT-4o`'s context window.

Task was to find all pairs of user IDs where both users have at least one "location" question and at least one "abbreviation" question.

Format: `(id1, id2)` with the lower ID first, no duplicates.

This is quadratic because:

- `1000` users = `499,500` possible pairs to check
- For each pair, you need to understand the semantic meaning of `~8` questions
- You can't use simple keyword matching (the questions are phrased naturally)

I computed the ground truth programmatically: `26,335` valid pairs.

Gave the whole thing to `GPT-4o` directly. No tricks, no scaffolding. Just the raw prompt.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": task_query + context}]
)
```

Results:

- **Found:** `379` pairs
- **Correct:** `194`
- **Missed:** `26,141` pairs (that's `99%` of them!)
- **Precision:** `51%` (half of what it found was wrong)
- **Recall:** `0.74%` (it found less than `1%` of valid pairs)
- **F1 Score:** `1.45%`

`GPT-4o`, collapsing on a task that fits comfortably in its context window.

The paper was right. Context rot + quadratic complexity = disaster.

## Trying to Build My Own RLM

Okay, so `GPT-4o` failed. Now let's see if the RLM approach actually works.

The concept seemed straightforward enough: instead of feeding the massive prompt directly to the model, store it as a Python variable, let the LLM write code to interact with it, and allow it to make recursive subcalls.

I tried two approaches.

### Building my own simple RLM

I started by implementing a minimal version of the RLM concept, just `20` lines of code:

```python
# Store context as a variable the LLM can access
env = {'context': context, 'llm_query': sub_llm_function}

# Give the LLM instructions
system_prompt = """You have a Python REPL with a 'context' variable 
and an llm_query() function. Write code to solve the task."""

# Let it write code, execute it, feed results back
for turn in range(10):
    llm_response = call_gpt4(conversation)
    if code_in_response:
        exec(code, env)  # Run the code
        # Feed execution results back to LLM
```

The LLM's first attempt tried to solve the problem with regex patterns:

```python
# It wrote this:
loc_patterns = [r'\bwhere\b', r'\bwhere is\b']
abbr_patterns = [r'\bstand for\b', r'\bshort for\b']
```

The problem? Our questions like `"What's the weather in Rio?"` don't match `r'\bwhere is\b'` exactly. The regex was too strict. Result: `0` pairs found.

I refined the prompt: *"You MUST use `llm_query()` to classify questions semantically. Code alone won't work."*

### Another attempt

The LLM got the message. It started making sub-LLM calls:

```python
for line in lines:  # 1000+ lines
    category = llm_query(f"Classify this: {line}")
```

One call per line. For `1000` users with `~4` questions each, that's `4,000` sub-LLM calls.

At `~2` seconds per call, that's over `2` hours of runtime.

After `20` minutes, I terminated it.

### Using the Official Library

The paper authors released an implementation. Maybe theirs would work better?

```python
from rlm import RLM

rlm = RLM(model="gpt-4o", recursive_model="gpt-4o-mini")
result = await rlm.acompletion(
    query="Find all pairs where BOTH users have location AND abbreviation questions.",
    context=context
)
```

Output:

> All valid user ID pairs have been displayed in required format.

But where are the pairs?

The library returned a summary instead of the actual data. I tried tweaking the prompt, adjusting parameters, checking the output format. Same result — summaries, no pairs.

Looking back, I realised a few things:

1. **The "Just Use Sub-Calls" Problem**

    When I told the model to use `llm_query()` liberally, it took me literally, making thousands of calls. The paper mentions this: different models have different "instincts." `GPT-4o` naturally batches work. Other models make hundreds or thousands.

    Without careful prompt engineering, you get either:

    - Too few sub-calls → fails (uses heuristics instead of semantic understanding)
    - Too many sub-calls → works but takes forever and costs a fortune

2. **The Output Format Challenge**

    Getting structured data out of an RLM is tricky. The library expects a `FINAL()` statement. But when the answer is above `20,000` pairs, does the model output all of them, or summarise? Turns out: it summarises.

3. **The Engineering Gap**

    The paper's results came from production implementations with:

    - Carefully tuned system prompts
    - Smart batching strategies
    - Proper output handling
    - Extensive testing

    My "20 lines of code" version? Not quite there.

## My Take

So I didn't get a perfect replication, but the experiment taught me something.

1. That `1.45%` F1 score proves the problem is real — `GPT-4o` collapsed on a `~23K` token task that fit easily in its `128K` context window, missing `99%` of the answer.

2. The core RLM insight is brilliant: instead of putting everything into the LLM hoping it remembers, treat the prompt as external data the model explores through code and recursive delegation — like giving someone a book and a computer instead of asking them to memorise it.

3. Implementation is a bit tricky. You need to balance subcall frequency (too few = fails, too many = expensive), engineer prompts carefully, and handle structured outputs properly. The paper's results came from months of work, not an afternoon hack lol.

4. Model "personality" matters — the paper showed `GPT-5` makes `~10` strategic sub-calls per task while `Qwen3-Coder` makes way more than that, meaning different models need different handling.

5. This isn't just academic: RLMs enable processing `10M+` token inputs and better performance on dense reasoning tasks using existing models, though with tradeoffs like cost variance and engineering complexity.

If you're working with large codebases, research papers, or legal documents where you can't skip information and relationships matter, RLMs are worth exploring — just don't expect it to work perfectly out of the box.
