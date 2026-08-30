# Local Language Models: An Overview

Small language models (SLMs) are models typically under 10 billion
parameters that are designed to run efficiently on consumer hardware such
as laptops, without requiring a dedicated GPU.

## Why run models locally

1. **Privacy** — data never leaves the device, which matters for sensitive
   documents, personal notes, or proprietary company data.
2. **Cost** — there is no per-token API charge; the only cost is the
   electricity to run the model.
3. **Offline availability** — the assistant works on a plane, in a basement
   server room, or anywhere without internet access.
4. **Latency** — for short prompts on capable hardware, local inference can
   be faster than a network round trip to a cloud API.

## Tradeoffs

Local, small models trade some reasoning capability and world knowledge for
these benefits. A 3B parameter model will generally hallucinate more and
follow complex multi-step instructions less reliably than a 70B+ parameter
cloud model. Retrieval-augmented generation (RAG) helps close this gap by
grounding answers in real documents instead of relying purely on the
model's internal knowledge.

## Quantization

Most local model deployments use quantized weights — for example, 4-bit
(Q4_K_M) instead of the original 16-bit floating point weights. This
reduces memory usage by roughly 4x with a modest, often barely noticeable,
quality tradeoff for everyday tasks.
