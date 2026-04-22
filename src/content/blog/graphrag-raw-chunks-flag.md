---
title: "How to View the Context GraphRAG Sends to Your LLM (New --raw-chunks Flag)"
description: "A new --raw-chunks flag I added to Microsoft GraphRAG that prints the exact raw context passed to the LLM during a query — for transparency, debugging, and quality assurance."
pubDate: 2025-04-23
readTime: "9 min read"
tags: ["graphrag", "rag", "llms", "microsoft", "open-source"]
---

![Cover image — GraphRAG --raw-chunks flag](/blog/graphrag-raw-chunks-flag/cover.webp)

Sometimes, when working with `RAG` systems, we need to know exactly what context is being passed to the LLM. You run a query, the model gives an answer, sometimes strange or irrelevant, and you are left wondering — "Did it actually get the right information? What was retrieved and sent to the model?"

Often, the issue isn't with the model itself, it's with the retrieved context from a database. Maybe irrelevant chunks were selected. Maybe important pieces were missing. To make this process more transparent, I added a simple but powerful feature to Microsoft GraphRAG: you can now print the raw context chunks that are sent to the LLM during a query. No additional configuration is required — the feature works out of the box. By default, the raw chunks are not displayed unless the `--raw-chunks` flag is explicitly provided, ensuring existing behavior remains unchanged.

```bash
graphrag query --method local --query "Do LLMs Struggle with Math Across Cultural Context" --root index --raw-chunks
```

With this `--raw-chunks` flag, GraphRAG will display not just the final answer, but also the raw retrieved context used to generate it. This gives you clear visibility into:

- What was retrieved
- What was passed to the LLM
- And why the model may have responded the way it did

It's a lightweight, non-intrusive addition — but it makes a big difference for debugging, development, and quality assurance.

## Code Changes and Implementation Details

To enable the `--raw-chunks` functionality in GraphRAG, I made modifications across six key files within the package. When you install GraphRAG using `pip install graphrag`, the full source code is downloaded into your environment. To implement this feature, you'll need to dive into the installed package and directly modify the relevant components.

The following files should be changed:

- `graphrag/query/factory.py`
- `graphrag/cli/main.py`
- `graphrag/cli/query.py`
- `graphrag/query/structured_search/local_search/search.py`
- `graphrag/query/structured_search/global_search/search.py`
- `graphrag/query/structured_search/drift_search/search.py`

Each of these files plays a role in the query flow and structured search logic. The `cli` module handles command-line parsing, so I updated it to accept the new `--raw-chunks` flag. In the `factory.py` and search-related modules, I integrated the logic needed to capture and return the raw retrieved context alongside the final output. No entirely new files, classes, or components were created — the feature was added by extending and modifying existing functions and classes within the current codebase. The implementation is designed to be non-intrusive: if the flag is not used, existing behavior remains unchanged.

### `factory.py` — adding `raw_chunks` parameter to factory functions

```python
# factory.py

def get_local_search_engine(
    reports: dict[str, list[CommunityReport]],
    text_units: dict[str, list[TextUnit]],
    # ... other parameters ...
    callbacks: list[QueryCallbacks] | None = None,
    raw_chunks: bool = True,  # Added parameter
) -> LocalSearch:
    """Create a local search engine based on data + configuration."""
    # ... existing setup code ...

    return LocalSearch(
        model=chat_model,
        system_prompt=system_prompt,
        context_builder=LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            # ... other parameters ...
        ),
        token_encoder=token_encoder,
        model_params=model_params,
        context_builder_params={
            # ... existing params ...
        },
        callbacks=callbacks,
        raw_chunks=raw_chunks,  # Pass the parameter
    )


def get_global_search_engine(
    # ... other parameters ...
    callbacks: list[QueryCallbacks] | None = None,
    raw_chunks: bool = True,  # Added parameter
) -> GlobalSearch:
    # ... existing code ...
    return GlobalSearch(
        # ... other parameters ...
        callbacks=callbacks,
        raw_chunks=raw_chunks,  # Pass the parameter
    )


def get_drift_search_engine(
    # ... other parameters ...
    callbacks: list[QueryCallbacks] | None = None,
    raw_chunks: bool = True,  # Added parameter
) -> DRIFTSearch:
    # ... existing code ...
    return DRIFTSearch(
        # ... other parameters ...
        callbacks=callbacks,
        raw_chunks=raw_chunks,  # Pass the parameter
    )
```

### `query.py` — adding raw chunks callback handling

```python
# query.py

class RawChunksCallback(QueryCallbacks):
    def on_context_chunk(self, chunk_type: str, chunk_data: Any):
        """Display raw chunks based on search type and chunk data."""
        if not chunk_data:
            return

        print(f"\n=== {chunk_type} ===")
        if isinstance(chunk_data, dict):
            for key, value in chunk_data.items():
                print(f"{key}:", value)
        elif isinstance(chunk_data, list):
            for item in chunk_data:
                print("- ", item)
        else:
            print(chunk_data)
        print("=" * 40)

    def on_map_reduce_chunk(self, stage: str, data: Any):
        """Display chunks from map-reduce process."""
        print(f"\n=== {stage} ===")
        print(data)
        print("=" * 40)
```

### `main.py` — adding CLI support for raw chunks

```python
# main.py

def _query_cli(
    query: str,
    search_type: str,
    config: GraphRagConfig,
    raw_chunks: bool = True,  # Added parameter
    # ... other parameters ...
) -> str:
    """Query the graph RAG system."""
    if search_type == "local":
        return run_local_search(
            query,
            config,
            raw_chunks=raw_chunks,  # Pass parameter
            # ... other parameters ...
        )
    elif search_type == "global":
        return run_global_search(
            query,
            config,
            raw_chunks=raw_chunks,  # Pass parameter
            # ... other parameters ...
        )
    elif search_type == "drift":
        return run_drift_search(
            query,
            config,
            raw_chunks=raw_chunks,  # Pass parameter
            # ... other parameters ...
        )


def main():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...
    parser.add_argument(
        "--raw-chunks",
        action="store_true",
        default=True,
        help="Show raw chunks retrieved from vector store",
    )
    args = parser.parse_args()

    response = _query_cli(
        args.query,
        args.search_type,
        config,
        raw_chunks=args.raw_chunks,  # Pass CLI argument
        # ... other parameters ...
    )
```

### Adding `raw_chunks` parameter to search functions

**Local Search changes:**

```python
# In LocalSearch constructor
class LocalSearch:
    def __init__(
        self,
        # ... other parameters ...
        raw_chunks: bool = True,  # New parameter
    ):
        self.raw_chunks = raw_chunks
        # ... rest of initialization ...

    def search(self, query: str) -> str:
        # Get context
        context = self.context_builder.build(query)

        # Show raw chunks if enabled
        if self.raw_chunks:
            print("\n=== Local Search Context ===")
            print("Text Units:", context.text_units)
            print("Community Reports:", context.community_reports)
            print("========================\n")

        # Process and return response
        return self._process_response(query, context)
```

**Global Search changes:**

```python
class GlobalSearch:
    def __init__(
        self,
        # ... other parameters ...
        raw_chunks: bool = True,
    ):
        self.raw_chunks = raw_chunks
        # ... rest of initialization ...

    def search(self, query: str) -> str:
        # Map phase
        map_responses = []
        for batch in self._get_batches():
            context = self.context_builder.build(query, batch)
            if self.raw_chunks:
                print(f"\n=== Map Phase Batch {len(map_responses)+1} ===")
                print("Context:", context)
                print("========================\n")
            response = self._map(query, context)
            map_responses.append(response)

        # Reduce phase
        if self.raw_chunks:
            print("\n=== Reduce Phase Context ===")
            print("Map Responses:", map_responses)
            print("========================\n")

        return self._reduce(query, map_responses)
```

**DRIFT Search changes:**

```python
class DRIFTSearch:
    def __init__(
        self,
        # ... other parameters ...
        raw_chunks: bool = True,
    ):
        self.raw_chunks = raw_chunks
        # ... rest of initialization ...

    def search(self, query: str) -> str:
        # Primer search
        primer_context = self._get_primer_context(query)
        if self.raw_chunks:
            print("\n=== DRIFT Primer Context ===")
            print("Query:", query)
            print("Context:", primer_context)
            print("========================\n")

        # Follow-up searches
        for epoch in range(self.n_depth):
            action_context = self._get_action_context(query, epoch)
            if self.raw_chunks:
                print(f"\n=== DRIFT Action Context (Epoch {epoch+1}) ===")
                print("Context:", action_context)
                print("========================\n")

        # Final synthesis
        final_context = self._get_final_context()
        if self.raw_chunks:
            print("\n=== DRIFT Final Synthesis Context ===")
            print("Context:", final_context)
            print("========================\n")

        return self._process_response(query, final_context)
```

## Testing

For the query:

```bash
graphrag query --method local --query "Do LLMs Struggle with Math Across Cultural Context" --root index
```

![Default GraphRAG query output — no raw chunks shown](/blog/graphrag-raw-chunks-flag/image_1.png)

For the same query with the new flag:

```bash
graphrag query --method local --query "Do LLMs Struggle with Math Across Cultural Context" --root index --raw-chunks
```

![GraphRAG query with --raw-chunks — raw retrieved context printed before the answer](/blog/graphrag-raw-chunks-flag/image_2.png)

![Continuation of --raw-chunks output showing the full retrieved context](/blog/graphrag-raw-chunks-flag/image_3.png)

## Current Status and Contribution

This feature is currently under review as a [pull request](https://github.com/microsoft/graphrag/pull/1886) to the main [Microsoft GraphRAG repository](https://github.com/microsoft/graphrag). Once the PR is reviewed and (hopefully) merged, the `--raw-chunks` flag will become a part of the official release. At that point, we'll simply be able to use it by upgrading our GraphRAG installation and adding `--raw-chunks` to your CLI query — no code modifications needed.

Until then, if we'd like to use this feature right away, we'll need to make manual changes to the files listed above. For convenience, I've included the updated code snippets inside a [Google Colab notebook](https://colab.research.google.com/drive/1futs5tlsSUlZN9ZDmfl3T-_GdeI8CHoG?usp=sharing), so we can easily copy and paste the relevant parts into your local setup.
