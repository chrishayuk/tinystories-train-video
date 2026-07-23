---
library_name: tokenizers
tags:
  - tokenizer
  - knowledge-first
language:
  - en
license: apache-2.0
---

# v11 — knowledge-first tokenizer

Vocabulary size: **71260**

v11 is a pure-Rust longest-match tokenizer whose vocabulary is assembled from knowledge sources (WordNet, Wikidata, tree-sitter AST grammars for 77 programming languages, curated morphemes, Greek letters, math symbols, acronyms) rather than discovered from BPE merges on a corpus.

Every token is a potential compilation target, so language keywords, scientific terms, acronyms, and Greek letters are guaranteed to be single pieces for efficient compilation.

## Load with HuggingFace
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
tok.encode("def fibonacci(n):").tokens
```

## Load with v11-cli
```bash
v11 encode --text 'def fibonacci(n):'
```
