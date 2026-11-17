# gpt2-inference
A "from scratch" NumPy based inference engine for GPT-2. This project is solely for me to improve my understanding of how inference works. I wanted to complete my own implementation of the LLM inference pipeline for GPT-2 starting from token embeddings through attention, MLP, token scoring, sampling, and then back out as tokens again.

There are two separate versions of this. The first (inference.py) is intentionally poorly optimized as the point here was helping me understand the concepts, and sometimes the vector math was simplifying the process to the point that I was missing some of the specifics on "what is this actually doing?" thus all of the wholly unnecessary for loops.

The second version (inference_optimized.py) took what was learned in the first version and then optimized it using matrix/vector math through NumPy to make the code about 15x faster on the prompt and 4.5x faster per generated token, and a little more concise.

## Setup Steps
```bash
pip install numpy tiktoken safetensors huggingface_hub
```

```bash
hf download openai-community/gpt2 config.json model.safetensors --local-dir gpt2
```

## Usage
Run `python inference.py` or `python inference_optimized.py`, type a prompt, and press Enter. The script streams the completion until it hits an end-of-text token or the 1024-token context limit.