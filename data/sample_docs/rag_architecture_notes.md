# RAG Architecture Notes

Retrieval-augmented generation (RAG) combines a retrieval system with a
generative language model. Instead of relying solely on knowledge baked
into the model's weights during training, RAG retrieves relevant text
snippets at query time and includes them in the prompt.

## Pipeline stages

1. **Ingestion** — documents are loaded and split into chunks of a few
   hundred characters, usually with some overlap between chunks so context
   isn't lost at chunk boundaries.
2. **Embedding** — each chunk is converted into a numerical vector using an
   embedding model. Semantically similar text produces vectors that are
   close together in vector space.
3. **Storage** — vectors are stored in a vector database (Chroma, FAISS,
   Qdrant, etc.) alongside the original text and metadata like the source
   filename.
4. **Retrieval** — at query time, the question is embedded with the same
   embedding model, and the vector database returns the most similar
   chunks by cosine similarity or another distance metric.
5. **Generation** — the retrieved chunks are inserted into the prompt as
   context, and the language model generates an answer grounded in that
   context rather than purely from memory.

## Common failure modes

- **Irrelevant retrieval**: chunks that are topically similar but don't
  actually answer the question. Mitigated by tuning chunk size and top-k.
- **Context window overflow**: too many or too long chunks pushed into the
  prompt, crowding out the model's ability to reason. Mitigated by
  reranking or summarizing retrieved chunks before insertion.
- **Stale index**: the vector database isn't updated when source documents
  change, leading to outdated answers.
