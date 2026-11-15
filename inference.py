import numpy as np
import tiktoken
import math
import json

from numpy import ndarray
from safetensors.numpy import load_file

class inference:
    model_type:str
    model_dimension:int
    model_layers:int
    model_heads:int
    layer_normal_epsilon:float
    residual_stream: ndarray
    context_window:int
    temperature = 0.7
    cache_k: list[ndarray]
    cache_v: list[ndarray]
    cache_len = 0
    w: dict[str, ndarray]

    def __init__(self, model:str):
        self.w = load_file(f"{model}/model.safetensors")

        with open(f"{model}/config.json", "r") as file:
            config = json.load(file)
            self.model_type = config["model_type"]
            self.model_dimension = config["n_embd"]
            self.model_layers = config["n_layer"]
            self.model_heads = config["n_head"]
            self.context_window = config["n_ctx"]
            self.layer_normal_epsilon = config["layer_norm_epsilon"]

        self.reset()

    def reset(self):
        self.cache_k = list()
        self.cache_v = list()
        for i in range(self.model_layers):
            self.cache_k.append(np.zeros((0, self.model_dimension), dtype=np.float32))
            self.cache_v.append(np.zeros((0, self.model_dimension), dtype=np.float32))
        self.cache_len = 0

    def invoke(self, token_ids:list[int], reset = False) -> int:

        if reset:
            self.reset()

        start = self.cache_len
        new_ids = token_ids[start:]
        n = len(new_ids)
        
        self.residual_stream = np.zeros((n, self.model_dimension), dtype=np.float32)

        self.cache_len = len(token_ids)

        #build the initial residual stream with the embedding and position weights for each token
        for i in range(n):
            for v in range(self.model_dimension):
                self.residual_stream[i][v] = self.w["wte.weight"][new_ids[i]][v] + self.w["wpe.weight"][start + i][v]

        for layer in range(self.model_layers):
            self.compute_layer(layer, start)

        norm_weights = self.w["ln_f.weight"]
        norm_bias = self.w["ln_f.bias"]

        normalized = self.normalize(self.residual_stream, norm_weights, norm_bias)

        last_token = normalized[-1,:]


        weighted_tokens = np.zeros(len(self.w["wte.weight"]))

        for i in range(len(weighted_tokens)):
            weighted_tokens[i] = np.dot(self.w["wte.weight"][i], last_token)

        softmaxed_tokens = self.softmax(weighted_tokens, self.temperature)

        top_ids = np.argsort(softmaxed_tokens)[-10:][::-1]   # 10 highest, descending
        top_probabilities = softmaxed_tokens[top_ids]
        top_probabilities = top_probabilities / top_probabilities.sum()
        next_id = int(np.random.choice(top_ids, p=top_probabilities))
            
        return next_id


    def compute_layer(self, layer:int, start:int):
        normalized_stream = self.normalize(self.residual_stream, self.w["h." + str(layer) + ".ln_1.weight"], self.w["h." + str(layer) + ".ln_1.bias"])

        attention_weights = self.w["h." + str(layer) + ".attn.c_attn.weight"]
        attention_bias = self.w["h." + str(layer) + ".attn.c_attn.bias"]

        q = np.zeros((len(self.residual_stream), self.model_dimension), dtype=np.float32)
        k = np.zeros((len(self.residual_stream), self.model_dimension), dtype=np.float32)
        v = np.zeros((len(self.residual_stream), self.model_dimension), dtype=np.float32)

        for t in range(len(self.residual_stream)):
            out = np.zeros(self.model_dimension * 3, dtype=np.float32)
            for j in range(self.model_dimension * 3):
                out[j] = np.dot(normalized_stream[t], attention_weights[:, j])   # one dot product per column
            out += attention_bias

            q[t] = out[0:self.model_dimension]
            k[t] = out[self.model_dimension:(self.model_dimension * 2)]
            v[t] = out[self.model_dimension * 2:(self.model_dimension * 3)]

        self.cache_k[layer] = np.concatenate((self.cache_k[layer], k))
        self.cache_v[layer] = np.concatenate((self.cache_v[layer], v))

        head_outputs = self.compute_attention(q, self.cache_k[layer], self.cache_v[layer], start)

        attention_projection_weights = self.w["h." + str(layer) + ".attn.c_proj.weight"]
        attention_projection_bias = self.w["h." + str(layer) + ".attn.c_proj.bias"]

        attention_out = head_outputs @ attention_projection_weights + attention_projection_bias
        self.residual_stream += attention_out

        mlp_out = self.compute_mlp(layer)
        self.residual_stream += mlp_out


    def normalize(self, x:ndarray, gamma:ndarray, beta:ndarray) -> ndarray:
        out = np.zeros_like(x)
        #for each token calculate the mean and the variance. Center mean on 0 and scale variance to 1
        for t in range(len(x)):
            row = x[t]
            mean = sum(x[t]) / self.model_dimension

            total_variance = 0
            for v in range(self.model_dimension):
                total_variance += (x[t][v] - mean) ** 2

            var = total_variance / self.model_dimension
            std = np.sqrt(var + 1e-5)

            for d in range(self.model_dimension):
                out[t, d] = (row[d] - mean) / std * gamma[d] + beta[d]
        return out

    def compute_attention(self, q:ndarray, k:ndarray, v:ndarray, start:int) -> ndarray:

        head_size:int = int(self.model_dimension / self.model_heads)
        outputs = np.zeros_like(q)
        score_scalar = math.sqrt(head_size)

        #at this point q,k,v are arrays of 768 for all tokens [tokens,768]. 768 is 64 values for each of the 12 heads
        for i in range(self.model_heads):
            head_start = i * head_size
            head_q = q[:, head_start:head_start + head_size]
            head_k = k[:, head_start:head_start + head_size]
            head_v = v[:, head_start:head_start + head_size]

            for t in range(len(head_q)):
                pos = start + t 
                scores = np.zeros(pos + 1, dtype=np.float32)

                for j in range(pos + 1):
                    scores[j] = np.dot(head_q[t], head_k[j]) / score_scalar

                weights = self.softmax(scores)

                weighted_sum = np.zeros(head_size, dtype=np.float32)

                for hv in range(head_size):
                    for x in range(len(scores)):
                        weighted_sum[hv] += head_v[x][hv] * weights[x]

                outputs[t, head_start:head_start + head_size] = weighted_sum
        return outputs

    def compute_mlp(self, layer:int) -> ndarray:
        normalization_weight = self.w[f"h.{layer}.ln_2.weight"]
        normalization_bias = self.w[f"h.{layer}.ln_2.bias"]
        feed_forward_weight = self.w[f"h.{layer}.mlp.c_fc.weight"]
        feed_forward_bias = self.w[f"h.{layer}.mlp.c_fc.bias"]
        projection_weight = self.w[f"h.{layer}.mlp.c_proj.weight"]
        projection_bias = self.w[f"h.{layer}.mlp.c_proj.bias"]

        normalized = self.normalize(self.residual_stream, normalization_weight, normalization_bias)

        hidden = normalized @ feed_forward_weight + feed_forward_bias
        hidden = self.gelu(hidden)
        mlp_out = hidden @ projection_weight + projection_bias

        return mlp_out


    def softmax(self, scores:ndarray, temperature:float = 1) -> ndarray:
        exps = np.exp((scores - scores.max()) / temperature)
        return exps / exps.sum()

    def gelu(self, x: ndarray) -> ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    
infer = inference("gpt2") 
enc = tiktoken.get_encoding(infer.model_type)

text = input()

token_ids = enc.encode(text)

print(enc.decode(token_ids), end="", flush=True)

for _ in range(infer.context_window):
    result = infer.invoke(token_ids)
    if result == enc.eot_token:
        break
    token_ids.append(result)

    print(enc.decode([result]), end="", flush=True)