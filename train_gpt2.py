import math
import torch 
from dataclasses import dataclass
from typing import cast, Tuple
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    

class CasualSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore[attr-defined]
        self.n_embed = config.n_embed
        self.n_head = config.n_head

        # bias here is the attention mask(ik, shitty naming)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) ==> (B, nh, T, T)
        mask = cast(Tensor, self.bias)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) ==> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y



class MLP(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        wte = cast(nn.Embedding, self.transformer.wte)
        wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 + self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tuple[Tensor, Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of size: {T}, block size is only: {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        wte = cast(nn.Embedding, self.transformer.wte)
        wpe = cast(nn.Embedding, self.transformer.wpe)
        tok_emb = wte(idx)
        pos_emb = wpe(pos)
        
        x = tok_emb + pos_emb

        h = cast(nn.ModuleList, self.transformer.h)
        for block in h:
            x = block(x)
        ln_f = cast(nn.LayerNorm, self.transformer.ln_f)
        x = ln_f(x)
        logits: Tensor = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    
    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys {sd_keys_hf} != {sd_keys}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                if sd_hf[k].shape != sd[k].shape:
                    print(k)
                    print("hf :", sd_hf[k].shape)
                    print("mine:", sd[k].shape)
                    raise ValueError("shape mismatch")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
model.eval()
model.to('cuda')

import tiktoken
enc = tiktoken.get_encoding('gpt2')


class DataLoaderLite:

    def __init__(self, B: int, T: int) -> None:
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens!!")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0

    def next_batch(self) -> Tuple[Tensor, Tensor]:   
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_pos += B * T

        if self.current_pos + (B * T + 1) > len(self.tokens):
            self.current_pos = 0 

        return x, y

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
train_loader = DataLoaderLite(4, 32)

model = GPT(GPTConfig())
model.to('cuda')
# logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x ,y = train_loader.next_batch()
    x, y = x.to(device='cuda'), y.to(device='cuda')
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step: {i}, Loss: {loss.item()}")

import sys; sys.exit(0)

# # prefix context
# tokens = enc.encode("hello! Im a language model")
# tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to('cuda')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, _ = model(x) # (B, T, vocab)
#         logits = logits[:, -1, :] # (B, vocab)
#         probs = F.softmax(logits, dim=-1)

#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(f"=> {decoded}")