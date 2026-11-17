import json
import math
import numpy as np
import tiktoken

from numpy import ndarray
from safetensors.numpy import load_file


class Inference:
    model_type: str
    model_dimension: int
    model_layers: int
    model_heads: int
    layer_norm_epsilon: float
    residual_stream: ndarray
    context_window: int
    temperature: float
    cache_k: list[ndarray]
    cache_v: list[ndarray]
    cache_len: int
    w: dict[str, ndarray]

    def __init__(self, model: str):
        self.w = load_file(f"{model}/model.safetensors")

        # Pull in model specific config settings
        with open(f"{model}/config.json") as file:
            config = json.load(file)
            self.model_type = config["model_type"]
            self.model_dimension = config["n_embd"]
            self.model_layers = config["n_layer"]
            self.model_heads = config["n_head"]
            self.context_window = config["n_ctx"]
            self.layer_norm_epsilon = config["layer_norm_epsilon"]
            self.temperature = 0.7

        self.reset()

    # reset the KV Cache either to set up a new instantiation, or to reset for a new prompt
    def reset(self):
        self.cache_k = list()
        self.cache_v = list()
        for _ in range(self.model_layers):
            self.cache_k.append(np.zeros((0, self.model_dimension), dtype=np.float32))
            self.cache_v.append(np.zeros((0, self.model_dimension), dtype=np.float32))
        self.cache_len = 0

    # provide a list of token ids to run inference against
    def invoke(self, token_ids: list[int], reset=False) -> int:
        if reset:
            self.reset()

        # Only process new tokens, relying on KV Cache for previously processed tokens
        start = self.cache_len
        new_ids = token_ids[start:]
        n = len(new_ids)
        self.cache_len = len(token_ids)

        # build the initial residual stream sized for the number of new tokens with the embedding and position weights for each token
        token_embeddings = self.w["wte.weight"]
        position_embeddings = self.w["wpe.weight"]

        self.residual_stream = (
            token_embeddings[new_ids] + position_embeddings[start : start + n]
        )

        # compute each layer with its results modifying the residual stream
        for layer in range(self.model_layers):
            self.compute_layer(layer, start)

        # one final normalization pass
        norm_weights = self.w["ln_f.weight"]
        norm_bias = self.w["ln_f.bias"]

        normalized = self.normalize(self.residual_stream, norm_weights, norm_bias)

        # next token scoring
        last_token = normalized[-1, :]
        logits = np.dot(token_embeddings, last_token)

        probabilities = self.softmax(logits, self.temperature)

        # sampling
        top_ids = np.argsort(probabilities)[-10:][::-1]  # 10 highest, descending
        top_probabilities = probabilities[top_ids]
        top_probabilities = top_probabilities / top_probabilities.sum()
        next_id = int(np.random.choice(top_ids, p=top_probabilities))

        return next_id

    # do all of the work to compute a single layer (attention/MLP)
    def compute_layer(self, layer: int, start: int):
        # pre-attention normalization
        pre_attention_normalization_weights = self.w[f"h.{layer}.ln_1.weight"]
        pre_attention_normalization_bias = self.w[f"h.{layer}.ln_1.bias"]
        normalized_stream = self.normalize(
            self.residual_stream,
            pre_attention_normalization_weights,
            pre_attention_normalization_bias,
        )

        # attention (query, key, value calculations)
        qkv_weights = self.w[f"h.{layer}.attn.c_attn.weight"]
        qkv_bias = self.w[f"h.{layer}.attn.c_attn.bias"]


        # QKV comes as a combined vector. In GPT2 this means 12 heads with 64 values for each Q,K, and V (2304 total)
        qkv = np.dot(normalized_stream, qkv_weights) + qkv_bias
        q,k,v = np.split(qkv, 3, axis=1)

        # add the new values to the KV Cache so we don't have to recalculate
        self.cache_k[layer] = np.concatenate((self.cache_k[layer], k))
        self.cache_v[layer] = np.concatenate((self.cache_v[layer], v))

        # take the q,k,v vectors and compute attention
        head_outputs = self.compute_attention(
            q, self.cache_k[layer], self.cache_v[layer], start
        )

        attention_projection_weights = self.w[f"h.{layer}.attn.c_proj.weight"]
        attention_projection_bias = self.w[f"h.{layer}.attn.c_proj.bias"]

        # project attention back onto the residual stream
        attention_out = (
            head_outputs @ attention_projection_weights + attention_projection_bias
        )
        self.residual_stream += attention_out

        # compute MLP for this layer and apply it to the residual stream
        mlp_out = self.compute_mlp(layer)
        self.residual_stream += mlp_out

    # normalize vectors to bring them back into a consistent space so we don't blow up the calculations
    def normalize(self, x: ndarray, gamma: ndarray, beta: ndarray) -> ndarray:
        # per-token: center to mean 0, scale to variance 1, then apply learned scale (gamma) and shift (beta)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.layer_norm_epsilon)
        out = (x - mean) / std * gamma + beta

        return out

    def compute_attention(
        self, q: ndarray, k: ndarray, v: ndarray, start: int
    ) -> ndarray:
        head_size: int = int(self.model_dimension / self.model_heads)
        outputs = np.zeros_like(q)
        score_scalar = math.sqrt(head_size)

        # at this point q,k,v are arrays of 768 for all tokens [tokens,768]. 768 is 64 values for each of the 12 heads
        # split q, k, v into per-head slices
        for i in range(self.model_heads):
            head_start = i * head_size
            head_q = q[:, head_start : head_start + head_size]
            head_k = k[:, head_start : head_start + head_size]
            head_v = v[:, head_start : head_start + head_size]

            # for each new token, score the key based on the query
            for t in range(len(head_q)):
                pos = start + t

                scores = np.dot(head_k[:pos + 1], head_q[t]) / score_scalar

                # softmax to scale the numbers as 0 to 1 and have them add up to 1.
                weights = self.softmax(scores)

                # calculate the weighted sum by multiplying the head value vectors by the attention weights
                weighted_sum = weights @ head_v[:pos + 1]

                # combine the resultant values per head back into a single vector per token
                outputs[t, head_start : head_start + head_size] = weighted_sum

        return outputs

    def compute_mlp(self, layer: int) -> ndarray:
        normalization_weight = self.w[f"h.{layer}.ln_2.weight"]
        normalization_bias = self.w[f"h.{layer}.ln_2.bias"]
        up_project_weight = self.w[f"h.{layer}.mlp.c_fc.weight"]
        up_project_bias = self.w[f"h.{layer}.mlp.c_fc.bias"]
        down_project_weight = self.w[f"h.{layer}.mlp.c_proj.weight"]
        down_project_bias = self.w[f"h.{layer}.mlp.c_proj.bias"]

        normalized = self.normalize(
            self.residual_stream, normalization_weight, normalization_bias
        )

        hidden = normalized @ up_project_weight + up_project_bias
        hidden = self.gelu(hidden)
        mlp_out = hidden @ down_project_weight + down_project_bias

        return mlp_out

    # scale values as 0 to 1 (with temperature) and make them add up to 1
    def softmax(self, scores: ndarray, temperature: float = 1) -> ndarray:
        exps = np.exp((scores - scores.max()) / temperature)
        return exps / exps.sum()

    # Gaussian Error Linear Unit
    def gelu(self, x: ndarray) -> ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


if __name__ == "__main__":
    infer = Inference("gpt2")
    enc = tiktoken.get_encoding("gpt2")

    text = input()

    token_ids = enc.encode(text)

    print(enc.decode(token_ids), end="", flush=True)

    while len(token_ids) < infer.context_window:
        result = infer.invoke(token_ids)
        if result == enc.eot_token:
            break
        token_ids.append(result)

        print(enc.decode([result]), end="", flush=True)
