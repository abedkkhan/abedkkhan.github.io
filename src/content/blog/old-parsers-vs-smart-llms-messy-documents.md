---
title: "Old Parsers vs. Smart LLMs: Which Understands Messy Documents Better?"
description: "Comparing traditional parsers like llmsherpa with GPT-4o for extracting structure and meaning from messy, complex documents — and why intelligent extraction matters for RAG pipelines."
pubDate: 2025-03-30
readTime: "8 min read"
tags: ["llms", "rag", "document-parsing", "gpt-4o", "graph-rag"]
---

![Cover image — old parsers vs smart LLMs for messy documents](/blog/old-parsers-vs-smart-llms-messy-documents/cover.webp)

We are surrounded by very messy data. Text is often unsupervised, unlabeled, with images scattered throughout documents, along with complex figures and tables that hold most of the actual information. The insight is stored less in words and more in visual or structured elements. At the same time, there's data in these documents that might not even be worth extracting like the table of contents or repetitive headers.

This mess of data is useless unless we extract it in a clean way without losing the actual semantic meaning or the true gist of the content. We need a structure. Each paragraph should connect with the next and the previous one. Extracted content from tables and figures should also maintain its context. Only then can we break down our data into meaningful chunks, create embeddings, form proper nodes and edges for Graph `RAG`, or use it for `RAG` pipelines in general. That's how we pretrain models with intelligent input — not just input and tokens, but smart tokens. So, we have two main approaches to extract text from documents:

1. Using traditional built-in parsing libraries like `llmsherpa`, and
2. Using LLMs.

## llmsherpa

`llmsherpa` library, which relies on `LayoutPDFReader` under the hood. These built-in parser libraries are pretty solid — they can extract text from input documents, and `llmsherpa` is considered one of the most powerful among them. That's because it doesn't just pull raw text, it parses PDFs along with hierarchical layout information like: sections and subsections, paragraphs, links between sections and paragraphs, tables along captions and headings, etc. But here's the issue, this works well only when the document itself is clean and properly structured. What happens when it gets a complex, messy document like below.

![A messy, real-world PDF that breaks traditional parsers](/blog/old-parsers-vs-smart-llms-messy-documents/image_1.png)

Source of the document: [PSP_Packages_Eligibility_and_Inclusions_FC_ITC.pdf](https://dcj.nsw.gov.au/documents/service-providers/out-of-home-care-and-permanency-support-program/contracts-funding-and-packages/PSP_Packages_Eligibility_and_Inclusions_FC_ITC.pdf)

- It fails to identify sections and subsections because there's no consistent formatting.
- It struggles with paragraphs, especially when the line breaks and indentation are irregular.
- And once the paragraphs and sections aren't clearly identified, it can't establish any meaningful links between them. Why? Because it just goes horizontally line by line, extracting text in a plain, mechanical way.

Same goes for tables. It does extract table content but many times, tables come in complex structures.

- Sometimes the information is stacked vertically, sometimes horizontally, sometimes even nested.
- But the parser just reads rows from left to right, without understanding the logical grouping of data.
- The output ends up being a bunch of disconnected lines of text — no structure, no meaning.

And that's the problem. If the parser doesn't understand the true layout and logic of the content, how are we supposed to create nodes and edges from it for Graph `RAG`? How do we get usable embeddings from something that's semantically broken? Example code and output of using `llmsherpa` to extract text from above document is following.

```python
import os
import sys
from llmsherpa.readers import LayoutPDFReader

llmsherpa_api_url = "http://localhost:5001/api/parseDocument?renderFormat=all"
pdf_url = ""
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

try:
    doc = pdf_reader.read_pdf(pdf_url)
    with open("output.txt", "w") as f:
        f.write(doc.to_text())
    print("\nText output saved to output.txt")
except Exception as e:
    print(f"Error processing PDF: {str(e)}")
```

![llmsherpa output — flat, disconnected text with no structure preserved](/blog/old-parsers-vs-smart-llms-messy-documents/image_2.png)

## LLMs

Let's talk about a different approach: using LLMs directly to parse documents — specifically `gpt-4o`, which comes with built-in OCR capabilities and the ability to understand and extract information from images.

To test this, we take the same complex, messy document. We convert the pages into `PNG` and pass them to `gpt-4o` using a carefully crafted prompt. The output? A structured representation of the content, far more aligned, coherent, and semantically rich than what we got using `llmsherpa`.

This method skips over the rigid parsing step and instead leans into the flexibility of LLMs. With `gpt-4o`, it's possible to analyze each page of a document regardless of its structure and extract meaningful metadata. Text, tables, figures, and even complex visuals can be processed while preserving the connections and context.

One of the biggest advantages is how LLMs handle tables and graphics. They don't just extract rows, they understand the logic. Whether the information is stacked vertically, horizontally, or embedded in visuals, `gpt-4o` can interpret and describe it. Even in cases where figures or charts don't contain text, the model is capable of explaining what the graphic represents and connecting it to surrounding content.

This approach leads to much cleaner, richer outputs — text that retains meaning, tables that are actually useful, and visuals that are explained instead of ignored. The extracted chunks are not only easier to work with but are also better suited for embeddings, `RAG` pipelines, or graph-based representations like Graph `RAG`. An example of the prompt used with `gpt-4o`, along with the extracted output from the same document, is shown below.

```python
class TableContent(BaseModel):
    """Table information with semantic context"""
    section: str = Field(description="Table section or category")
    content: Dict[str, List[str]] = Field(description="Structured table content")
    context: str = Field(description="Table significance and relationships")

class ExtractedContent(BaseModel):
    """Document content with semantic preservation"""
    title: str = Field(description="Document title or main heading")
    raw_text: str = Field(description="Complete raw text from the image in paragraph format")
    main_content: List[str] = Field(description="Key content sections")
    table_content: List[TableContent] = Field(description="Structured table data")
    technical_terms: List[str] = Field(description="Technical terminology")
    visual_description: str = Field(description="Visual element description")
    summary: str = Field(description="Contextual summary")


response = client.chat.completions.create(
    model="openai/chatgpt-4o-latest",
    messages=[
        {
            "role": "system",
            "content": """Analyze document images with focus on
                           complete content extraction:
            1. Raw Text Extraction:
               - Extract ALL text exactly as it appears in the image
               - Maintain paragraph structure and formatting
               - Include ALL headers, labels, and annotations
               - Preserve text order and hierarchy
            2. Document Structure:
               - Extract title and main headings
               - Identify key content sections
               - Maintain document hierarchy
            3. Table Analysis:
               - Carefully read and understand tables
               - Preserve relationships between data
               - Convert tables to meaningful text
               - Explain table context and significance
            4. Visual Analysis:
               - Describe diagrams and flowcharts
               - Explain visual relationships
               - Provide context for images
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract ALL text from this image exactly as it appears, "
                            "maintaining formatting and structure. Also analyze tables and "
                            "visual elements."
                },
            ]
        }
    ]
)
```

![GPT-4o output — structured, context-preserving extraction from the same messy document](/blog/old-parsers-vs-smart-llms-messy-documents/image_3.png)

There's a clear difference between the two extracted outputs and it's obvious that `gpt-4o` performs intelligent extraction. With LLMs, the prompt can be tweaked based on the structure of the input document, giving full control over how the content is extracted. We are not just extracting text — we are extracting intelligent tokens, preserving meaning, context, and structure. This kind of extraction might not be possible with traditional parsers. It's the LLM that truly understands and extracts the data the way we need it.
