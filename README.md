# gpt2-inference
A "from scratch" NumPy based inference engine for GPT2. This project is solely for me to improve my understanding of how inference works. I wanted to complete my own implementation of the LLM inference pipeline for GPT2 starting from tokenization and embeddings through attention, MLP, token scoring, sampling, and then back out as tokens again.

This is intentionally poorly optimized as the point here was helping me understand the concepts, and sometimes the vector math was simplifying the process to the point that I was missing some of the specifics on "what is this actually doing?" thus all of the wholly unnecessary for loops.

## Setup Steps
```bash
pip install numpy tiktoken safetensors huggingface_hub
```

```bash
hf download openai-community/gpt2 config.json model.safetensors --local-dir gpt2
```

## Usage
Run `python inference.py`, type a prompt, and press Enter. The script streams the completion until it hits an end-of-text token or the 1024-token context limit.